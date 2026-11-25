/** An OpenAI-style function tool declaration, as `tools` takes it. */
export interface FunctionTool {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters?: unknown
  }
}

export type ParsedTools = { tools: FunctionTool[] } | { error: string }

/**
 * The function tools in `text`: a JSON array of OpenAI-style declarations
 * (`[{"type": "function", "function": {"name": ..., "parameters": ...}}]`), or why it isn't
 * one. Blank text is no tools.
 */
export function parseTools(text: string): ParsedTools {
  if (text.trim() === '') {
    return { tools: [] }
  }
  let json: unknown
  try {
    json = JSON.parse(text)
  } catch (e) {
    return { error: `the tools are not JSON: ${(e as Error).message}` }
  }
  if (!Array.isArray(json)) {
    return { error: 'the tools must be a JSON array of tools' }
  }
  const tools: FunctionTool[] = []
  for (const [i, t] of json.entries()) {
    const f = t?.type === 'function' ? t.function : undefined
    if (typeof f !== 'object' || f === null) {
      return { error: `tools[${i}] is not a function tool` }
    }
    if (typeof f.name !== 'string' || f.name === '') {
      return { error: `tools[${i}].function.name must be a string` }
    }
    tools.push({
      type: 'function',
      function: {
        name: f.name,
        ...(typeof f.description === 'string' ? { description: f.description } : {}),
        ...(f.parameters != null ? { parameters: f.parameters } : {}),
      },
    })
  }
  return { tools }
}
