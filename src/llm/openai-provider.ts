/**
 * OpenAI Provider – Chat Completions with function calling / tool_calls support
 */
import type {
  LLMProvider,
  Message,
  ToolDefinition,
  LLMResponse,
  LLMOptions,
  ToolCall,
} from './types';

const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';
const DEFAULT_MODEL = 'gpt-5.2';
const DEFAULT_MAX_TOKENS = 8192;

// --- GIGACHAT AUTH HELPER ---
let cachedGigaToken: string | null = null;
let gigaTokenExpiry: number = 0;

async function getGigaChatToken(authKey: string): Promise<string> {
  const now = Date.now();
  // Если токен есть и до его истечения больше 1 минуты, используем его
  if (cachedGigaToken && now < gigaTokenExpiry - 60000) {
    return cachedGigaToken;
  }

  const response = await fetch('https://ngw.devices.sberbank.ru:9443/api/v2/oauth', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json',
      'RqUID': '6f0b1291-b6c9-4d5c-8e3a-8c5d4e4f7b6a', // Случайный UUID
      'Authorization': `Basic ${authKey}`
    },
    body: 'scope=GIGACHAT_API_PERS' // Для физических лиц
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`GigaChat Auth Error: ${response.status} - ${err}`);
  }

  const data = await response.json();
  cachedGigaToken = data.access_token;
  gigaTokenExpiry = data.expires_at; // Время истечения в мс
  
  return cachedGigaToken!;
}
// ----------------------------

export class OpenAIProvider implements LLMProvider {
  private apiKey: string;
  private defaultModel: string;
  private baseUrl: string;

  constructor(
    apiKey: string,
    defaultModel: string = DEFAULT_MODEL,
    baseUrl: string = OPENAI_API_URL,
  ) {
    this.apiKey = apiKey;
    this.defaultModel = defaultModel;
    this.baseUrl = baseUrl;
  }

  getCurrentModel(): string {
    return this.defaultModel;
  }

  /** Вспомогательный метод для получения правильного ключа (обычный или GigaChat) */
  private async getAuthKey(): Promise<string> {
    if (this.baseUrl.includes('giga.chat')) {
      return await getGigaChatToken(this.apiKey);
    }
    return this.apiKey;
  }

  /** Test the connection by sending a minimal request */
  async testConnection(): Promise<{
    success: boolean;
    error?: string;
    response?: string;
  }> {
    try {
      const actualKey = await this.getAuthKey();

      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${actualKey}`,
        },
        body: JSON.stringify({
          model: this.defaultModel,
          messages: [{ role: 'user', content: 'test' }],
        }),
      }); 

      const rawText = await response.text();

      if (!response.ok) {
        return {
          success: false,
          error: `HTTP ${response.status}: ${rawText.slice(0, 200)}`,
        };
      }

      // Parse JSON and verify structure
      let data: unknown;
      try {
        data = JSON.parse(rawText);
      } catch {
        return {
          success: false,
          error: `Invalid JSON response: ${rawText.slice(0, 200)}`,
        };
      }

      // Check for minimal valid structure
      const typedData = data as {
        choices?: Array<{ message?: { content?: string } }>;
      };
      if (
        !typedData.choices ||
        !typedData.choices[0] ||
        !typedData.choices[0].message?.content
      ) {
        return {
          success: false,
          error: 'Invalid response structure: missing choices.message.content',
        };
      }

      return {
        success: true,
        response: typedData.choices[0].message.content,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  async chat(
    messages: Message[],
    tools: ToolDefinition[],
    options: LLMOptions = {},
  ): Promise<LLMResponse> {
    // OpenAI uses the same role structure, just map tool messages correctly
    const openAIMessages = messages.map(msg => {
      if (msg.role === 'tool') {
        return {
          role: 'tool' as const,
          content: msg.content,
          tool_call_id: msg.toolCallId ?? '',
        };
      }
      if (
        msg.role === 'assistant' &&
        msg.toolCalls &&
        msg.toolCalls.length > 0
      ) {
        return {
          role: 'assistant' as const,
          content: msg.content || null,
          tool_calls: msg.toolCalls.map(tc => ({
            id: tc.id,
            type: 'function' as const,
            function: {
              name: tc.name,
              arguments: JSON.stringify(tc.arguments),
            },
          })),
        };
      }
      return {
        role: msg.role as 'system' | 'user' | 'assistant',
        content: msg.content,
      };
    });

    const resolvedModel = this.defaultModel;

    // Newer OpenAI models (gpt-4.1+, gpt-5+, o-series) require
    // 'max_completion_tokens' instead of the legacy 'max_tokens' parameter.
    const useNewTokenParam = /^(gpt-4\.[1-9]|gpt-[5-9]|o[1-9])/.test(
      this.defaultModel,
    );
    const tokenLimit = options.maxTokens ?? DEFAULT_MAX_TOKENS;

    const body: Record<string, unknown> = {
      model: resolvedModel,
      [useNewTokenParam ? 'max_completion_tokens' : 'max_tokens']: tokenLimit,
      messages: openAIMessages,
    };

    if (tools.length > 0) {
      body.tools = tools;
      body.tool_choice = 'auto';
    }
    if (options.temperature !== undefined) {
      body.temperature = options.temperature;
    }

    // Получаем актуальный токен перед запросом
    const actualKey = await this.getAuthKey();

    const response = await fetch(this.baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${actualKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      // Try to extract a concise error message from the JSON payload
      let detail = errorBody;
      try {
        const parsed = JSON.parse(errorBody) as {
          error?: { message?: string; code?: string; type?: string };
        };
        if (parsed.error?.message) {
          detail = parsed.error.code
            ? `${parsed.error.code}: ${parsed.error.message}`
            : parsed.error.message;
        }
      } catch {
        // Not JSON – use raw body
      }
      throw new Error(`OpenAI API error ${response.status}: ${detail}`);
    }

    const rawText = await response.text();
    let data: {
      choices: {
        message: {
          content: string | null;
          tool_calls?: {
            id: string;
            type: string;
            function: { name: string; arguments: string };
          }[];
        };
        finish_reason: string;
      }[];
      usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
      };
    };
    try {
      data = JSON.parse(rawText) as typeof data;
    } catch (parseErr) {
      const contentType = response.headers.get('content-type') ?? 'unknown';
      throw new Error(
        `Invalid JSON response from custom LLM endpoint\n` +
          `URL: ${this.baseUrl}\n` +
          `Status: ${response.status}\n` +
          `Content-Type: ${contentType}\n` +
          `Raw response (first 500 chars): ${rawText.slice(0, 500)}`,
      );
    }

    const choice = data.choices[0];
    const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).map(tc => {
      let argumentsObj: Record<string, unknown>;
      try {
        argumentsObj = JSON.parse(tc.function.arguments) as Record<
          string,
          unknown
        >;
      } catch (parseErr) {
        throw new Error(
          `Invalid JSON in tool call arguments for "${tc.function.name}"\n` +
            `Raw arguments: ${tc.function.arguments}`,
        );
      }
      return {
        id: tc.id,
        type: 'function' as const,
        name: tc.function.name,
        arguments: argumentsObj,
      };
    });

    const finishReason =
      choice.finish_reason === 'tool_calls'
        ? 'tool_calls'
        : choice.finish_reason === 'stop'
        ? 'stop'
        : 'stop';

    return {
      content: choice.message.content ?? '',
      toolCalls,
      finishReason,
      usage: {
        promptTokens: data.usage.prompt_tokens,
        completionTokens: data.usage.completion_tokens,
        totalTokens: data.usage.total_tokens,
      },
    };
  }
}
