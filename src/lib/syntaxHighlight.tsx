import React from 'react';

interface Token {
  type: 'comment' | 'string' | 'keyword' | 'type' | 'number' | 'function' | 'decorator' | 'plain';
  value: string;
}

const TOKEN_REGEX =
  /(\/\/[^\n]*|#[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(@[a-zA-Z_]\w*|\b(?:export|import|from|const|let|var|function|return|if|else|for|while|switch|case|break|continue|new|class|interface|type|extends|implements|async|await|try|catch|finally|throw|def|elif|lambda|yield|with|as|pass|raise|in|is|not|and|or|True|False|None|null|undefined|true|false|void|int|float|double|char|bool|unsigned|long|short|struct|enum|union|typedef|static|inline|extern|volatile|register|sizeof|namespace|template|typename|public|private|protected|virtual|override|final|package|func|chan|defer|go|select|range|map|fallthrough|uniform|varying|attribute|in|out|inout|precision|highp|mediump|lowp|vec2|vec3|vec4|mat2|mat3|mat4|sampler2D|gl_FragColor|gl_Position|__global__|__device__|__shared__|__host__|__constant__|__restrict__|SELECT|FROM|WHERE|INSERT|INTO|UPDATE|DELETE|JOIN|LEFT|RIGHT|INNER|ON|GROUP|BY|ORDER|LIMIT|CREATE|TABLE|INDEX|POLICY|USING)\b)|(\b(?:string|number|boolean|any|unknown|never|Record|Promise|Array|Readonly|Partial|React|Tensor|torch|triton|tl|np|cuda|uint8_t|uint16_t|uint32_t|uint64_t|int32_t|int64_t|size_t|half|__nv_fp8_e4m3)\b)|(\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?f?)\b)|(\b[a-zA-Z_]\w*(?=\s*\())/g;

function tokenizeCode(code: string): Token[] {
  const tokens: Token[] = [];
  let lastIndex = 0;

  for (const match of code.matchAll(TOKEN_REGEX)) {
    const matchIndex = match.index ?? 0;
    if (matchIndex > lastIndex) {
      tokens.push({
        type: 'plain',
        value: code.slice(lastIndex, matchIndex),
      });
    }

    const [full, comment, str, decorator, keyword, typeName, num, fn] = match;
    if (comment !== undefined) {
      tokens.push({ type: 'comment', value: full });
    } else if (str !== undefined) {
      tokens.push({ type: 'string', value: full });
    } else if (decorator !== undefined) {
      tokens.push({ type: 'decorator', value: full });
    } else if (keyword !== undefined) {
      tokens.push({ type: 'keyword', value: full });
    } else if (typeName !== undefined) {
      tokens.push({ type: 'type', value: full });
    } else if (num !== undefined) {
      tokens.push({ type: 'number', value: full });
    } else if (fn !== undefined) {
      tokens.push({ type: 'function', value: full });
    } else {
      tokens.push({ type: 'plain', value: full });
    }

    lastIndex = matchIndex + full.length;
  }

  if (lastIndex < code.length) {
    tokens.push({
      type: 'plain',
      value: code.slice(lastIndex),
    });
  }

  return tokens;
}

const TOKEN_CLASSES: Record<Token['type'], string> = {
  comment: 'text-slate-500 italic',
  string: 'text-emerald-300',
  keyword: 'text-purple-300 font-semibold',
  type: 'text-[#80ffdb]',
  number: 'text-amber-300',
  function: 'text-[#64dfdf]',
  decorator: 'text-pink-400',
  plain: 'text-slate-100',
};

export function renderHighlightedCode(code: string): React.ReactNode {
  const tokens = tokenizeCode(code);
  return tokens.map((tok, i) => {
    if (tok.type === 'plain') {
      return <React.Fragment key={i}>{tok.value}</React.Fragment>;
    }
    return (
      <span key={i} className={TOKEN_CLASSES[tok.type]}>
        {tok.value}
      </span>
    );
  });
}

export function slugifyHeading(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`*_~]/g, '')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-');
}
