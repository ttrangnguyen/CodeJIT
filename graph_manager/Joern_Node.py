def clean_code(code):
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
        '(', ')', '[', ']', ";", ",", "_",
        '{', '}'
    }

    for item in operators1:
        code = code.replace(item, " ")
    for item in operators2:
        code = code.replace(item, " ")
    for item in operators3:
        code = code.replace(item, " ")
    return code.lower()

def print_stmt(stmt_nodes, edges):
    #print(stmt_nodes.columns)
    sentence = ""
    for i, n in stmt_nodes.iterrows():
        #print(n)
        tmp = NODE(stmt_nodes.at[i, "id"], stmt_nodes.at[i, "_label"], stmt_nodes.at[i, "code"],
                   stmt_nodes.at[i, "name"])
        sentence += tmp.print_node()
    return sentence

def get_list_stmts(stmt_id_list, operation_ctxs):
  content = " "
  for x in stmt_id_list:
    content += operation_ctxs[x] + " "
  return content

class NODE:
    def __init__(self, _id, label, code, name):

        self.id = _id
        if isinstance(label, str):
            self.label = label
        else:
            self.label = ""
        if isinstance(code, str):
            self.code = code.replace("\n", " ")
            self.code = clean_code(self.code)
        else:
            self.code = ""
        if isinstance(name, str):
            self.name = name.replace("\n", " ")
        else:
            self.name = ""
        # self.parents = edges[edges["innode"] == _id]["outnode"].tolist()
        # self.children = edges[edges["outnode"] == _id]["innode"].tolist()
        # print(self.parents)

    def print_node(self):
        result = ""
        if self.label == "META_DATA":
            result = ""
        elif self.label == "CONTROL_STRUCTURE" and self.code != "":
            if "if" in self.code:
                result = "ifStmt "
            elif "while" in self.code:
                result = "whileStmt "
            elif "for" in self.code:
                result = "forStmt "
            else:
                result = "CtrlStruct "
        elif "operator" in self.name:
            result = self.name.replace("<operator>.", "") + " "
        elif self.label == "CALL":
            result = "funcCall " + self.name + " "
            #result = "funcCall "
        elif self.label == "IDENTIFIER" or self.label == "METHOD_PARAMETER_IN" or self.label == "METHOD_PARAMETER_OUT" or self.label == "FIELD_IDENTIFIER" or self.label == "LOCAL":
            result = "variable " + self.name + " "
            #result = "variable "
        elif self.label == "METHOD_RETURN" or self.label == "RETURN":
            if self.name != "":
                result = "return " + self.name + " "
            else:
                result = "return " + self.code + " "
            #result = "return "
        elif self.label == "NAMESPACE":
            result = "namespace "
        elif self.label == "COMMENT":
            result = "comment "
        elif self.label == "FILE":
            result = "file "
        elif self.label == "LITERAL":
            result = "literal " + self.name + " "
            #result = "literal "
        elif self.label == "UNKNOWN":
            result = self.code + " "
        elif self.label == "TYPE" or self.label == "TYPE_DECL":
            result = "type " + self.name + " "
            #result = "type "
        elif self.label == "METHOD":
            if self.name != "<global>":
                result = "methodDecl " + self.name + " "
                #result = "methodDecl "
        elif self.label != "BLOCK":
            result = self.label + " "
        #result += self.code + " "
        return result
