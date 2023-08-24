CTX_SURROUNDING = "ctx_surrounding"
CTX_CDG = "ctx_cfg"
CTX_DDG = "ctx_ddg"
CTX_OPERATION = "ctx_operation"
VUL_TYPE = "vul_type"

# Sets for operators
operators1 = {
    '->', '++', '--',
    '<<=', '>>=', 
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|=', "::"
}
operators2 = {"!", "+", "-", "*", "/", "%", "<", ">",
    "&", "^", "?", "=", ":"}
operators3 = {
    '(', ')', '[', ']', ";",
    '{', '}'
}

"""
Takes a line of C++ code (string) as input
Tokenizes C++ code (breaks down into identifier, variables, keywords, operators)
Returns a list of tokens, preserving order in which they appear
"""

def tokenize(line):
    return line.split()


def clean_gadget(gadget):
  if not isinstance(gadget, str):
    return ""
  for item in operators1:
    gadget = gadget.replace(item, " " + item + " ")
  for item in operators2:
    gadget = gadget.replace(item, " " + item + " ")
  for item in operators3:
    gadget = gadget.replace(item, " ")
  return gadget.lower()

def get_operation_context(operation_ctx):
  context = operation_ctx
  return clean_gadget(context)
  
def get_context(pred, succ):
  surrounding_context = ""
  
  if isinstance(pred, str):
    surrounding_context += pred
  
  if isinstance(succ, str):
    surrounding_context += succ
  surrounding_context = clean_gadget(surrounding_context)
  return surrounding_context

def get_pre_context(pre_context):
  if(not isinstance(pre_context, str)):
    return "NONE"
  tmp = ""
  ctx = pre_context.split("\n")
  count = 0
  ctx_idx = len(ctx) - 1
  while(count < 10 and ctx_idx > 0):
    tmp  = ctx[ctx_idx] + " " + tmp
    count += 1
    ctx_idx -= 1
  return clean_gadget(tmp) 

def get_succ_context(succ_context):
  if(not isinstance(succ_context, str)):
    return "NONE"
  tmp = ""
  ctx = succ_context.split("\n")
  count = 0
  
  while(count < 10 and count < len(ctx)):
    tmp  += " " + ctx[count] 
    count += 1

  return clean_gadget(tmp) 