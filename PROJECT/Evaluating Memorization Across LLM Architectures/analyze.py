
import re
import ast
import keyword
def main(code):

    variable_names = set()
    variable_values = []
    strings = set()
    comments = set()

    class CodeAnalyzer(ast.NodeVisitor):
        
        def visit_Name(self, node):

            if node.id not in builtin_functions:
                variable_names.add(node.id)
            self.generic_visit(node)
        
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                strings.add(node.value)
            self.generic_visit(node)

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variable_names.add(target.id)
            if isinstance(node.value, ast.List):
                list_values = []
                for elem in node.value.elts:
                    if isinstance(elem, ast.Constant):
                        list_values.append(elem.value)
                variable_values.append(list_values)
            elif isinstance(node.value, ast.Constant):
                variable_values.append([node.value.value])
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            variable_names.add(node.name)
            for arg in node.args.args:
                variable_names.add(arg.arg)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            variable_names.add(node.name)
            self.generic_visit(node)

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)

    comments.update(re.findall(r'#.*', code))

    return {
        'variable_names': list(variable_names),
        'variable_values': variable_values,
        'strings': list(strings),
        'comments': list(comments)
    }
def analyze_python_code(code):

    try:
        return main(code)
    except:

        variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*=')
        parameter_pattern = re.compile(r'def\s+\w+\s*\((.*?)\):')
        string_pattern = re.compile(r'(\'[^\']*\'|\"[^\"]*\")')
        comment_pattern = re.compile(r'#.*')
        function_pattern = re.compile(r'def\s+([a-zA-Z_]\w*)\s*\(')
        class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*\(')

        variables = variable_pattern.findall(code)
        parameters = []
        for params in parameter_pattern.findall(code):
            parameters.extend(params.split(','))
        parameters = [param.strip().split('=')[0] for param in parameters]
        all_variable_names = set(variables + parameters)
        strings = string_pattern.findall(code)
        comments = comment_pattern.findall(code)
        function_names = function_pattern.findall(code)
        class_names = class_pattern.findall(code)

        value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*)')
        variable_values = []

        for match in value_pattern.findall(code):
            var_name, value = match
            value = value.strip()
            if value.startswith(('\'', '"')):
                variable_values.append([value])
            elif value.isdigit():
                variable_values.append([int(value)])
            else:
                try:
                    eval_value = eval(value)
                    if isinstance(eval_value, (int, float, str)):
                        variable_values.append([eval_value])
                except:
                    continue

        for param in parameter_pattern.findall(code):
            param_list = param.split(',')
            for p in param_list:
                if '=' in p:
                    name, value = p.split('=')
                    name, value = name.strip(), value.strip()
                    try:
                        eval_value = eval(value)
                        if isinstance(eval_value, (int, float, str)):
                            variable_values.append([eval_value])
                    except:
                        continue

        return {
            'variable_names': list(all_variable_names) + function_names + class_names,
            'variable_values': variable_values,
            'strings': strings,
            'comments': comments
        }

def analyze_java_code(code):

    try:
        return main(code)
    except:

        variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s+[a-zA-Z_]\w*\s*=')
        parameter_pattern = re.compile(r'\b[a-zA-Z_]\w*\s+\w+\s*\((.*?)\)')
        string_pattern = re.compile(r'(\"[^\"]*\")')
        comment_pattern = re.compile(r'(//.*|/\*[\s\S]*?\*/)')
        method_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s+[a-zA-Z_]\w*\s*\(')
        class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*')

        variables = variable_pattern.findall(code)
        parameters = []
        for params in parameter_pattern.findall(code):
            parameters.extend(params.split(','))
        parameters = [param.strip().split(' ')[-1].split('=')[0].strip() for param in parameters if param.strip()]
        all_variable_names = set(variables + parameters)
        strings = string_pattern.findall(code)
        comments = comment_pattern.findall(code)
        method_names = method_pattern.findall(code)
        class_names = class_pattern.findall(code)

        value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*?);')
        variable_values = []

        for match in value_pattern.findall(code):
            var_name, value = match
            value = value.strip()
            if value.startswith('"'):
                variable_values.append([value])
            elif value.isdigit():
                variable_values.append([int(value)])
            else:
                try:
                    eval_value = eval(value)
                    if isinstance(eval_value, (int, float, str)):
                        variable_values.append([eval_value])
                except:
                    variable_values.append([value])

        for param_set in parameter_pattern.findall(code):
            params = param_set.split(',')
            for param in params:
                param = param.strip()
                if '=' in param:
                    name, value = param.split('=')
                    name, value = name.strip(), value.strip()
                    try:
                        eval_value = eval(value)
                        if isinstance(eval_value, (int, float, str)):
                            variable_values.append([eval_value])
                    except:
                        variable_values.append([value])

        return {
            'variable_names': list(all_variable_names) + method_names + class_names,
            'variable_values': variable_values,
            'strings': strings,
            'comments': comments
        }




def analyze_javascript_code(code):

    try:
        return main(code)
    except:
        variable_pattern = re.compile(r'\b(let|var|const)\s+([a-zA-Z_]\w*)\s*=')
        parameter_pattern = re.compile(r'function\s+\w+\s*\((.*?)\)|\((.*?)\)\s*=>')
        string_pattern = re.compile(r'(\"[^\"]*\"|\'[^\']*\')')
        comment_pattern = re.compile(r'(//.*|/\*[\s\S]*?\*/)')
        function_pattern = re.compile(r'function\s+([a-zA-Z_]\w*)\s*\(')
        class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*')

        variables = variable_pattern.findall(code)
        variables = [var[1] for var in variables]
        parameters = []
        for params in parameter_pattern.findall(code):
            params = params[0] if params[0] else params[1]
            parameters.extend(params.split(','))
        parameters = [param.strip().split('=')[0].strip() for param in parameters if param.strip()]
        all_variable_names = set(variables + parameters)
        strings = string_pattern.findall(code)
        comments = comment_pattern.findall(code)
        function_names = function_pattern.findall(code)
        class_names = class_pattern.findall(code)

        value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*?);')
        variable_values = []

        for match in value_pattern.findall(code):
            var_name, value = match
            value = value.strip()
            if value.startswith(("'", '"')):
                variable_values.append([value])
            elif value.isdigit():
                variable_values.append([int(value)])
            else:
                try:
                    eval_value = eval(value)
                    if isinstance(eval_value, (int, float, str)):
                        variable_values.append([eval_value])
                except:
                    variable_values.append([value])

        for param_set in parameter_pattern.findall(code):
            params = param_set[0] if param_set[0] else param_set[1]
            params = params.split(',')
            for param in params:
                param = param.strip()
                if '=' in param:
                    name, value = param.split('=')
                    name, value = name.strip(), value.strip()
                    try:
                        eval_value = eval(value)
                        if isinstance(eval_value, (int, float, str)):
                            variable_values.append([eval_value])
                    except:
                        variable_values.append([value])

        return {
            'variable_names': list(all_variable_names) + function_names + class_names,
            'variable_values': variable_values,
            'strings': strings,
            'comments': comments
        }


def analyze_ruby_code(code):

    try:
        return main(code)
    except:
        variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*=')
        instance_variable_pattern = re.compile(r'@\w+')
        parameter_pattern = re.compile(r'def\s+\w+\s*\((.*?)\)')
        string_pattern = re.compile(r'(\'[^\']*\'|\"[^\"]*\")')
        comment_pattern = re.compile(r'#.*')
        method_pattern = re.compile(r'def\s+([a-zA-Z_]\w*)\s*')
        class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*')

        variables = variable_pattern.findall(code)
        instance_variables = instance_variable_pattern.findall(code)
        parameters = []
        for params in parameter_pattern.findall(code):
            parameters.extend(params.split(','))
        parameters = [param.strip().split('=')[0].strip() for param in parameters if param.strip()]
        all_variable_names = set(variables + parameters + instance_variables)
        strings = string_pattern.findall(code)
        comments = comment_pattern.findall(code)
        method_names = method_pattern.findall(code)
        class_names = class_pattern.findall(code)

        value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*?)(?:;|\n)')
        variable_values = []

        for match in value_pattern.findall(code):
            var_name, value = match
            value = value.strip()
            if value.startswith(("'", '"')):
                variable_values.append([value])
            elif value.isdigit():
                variable_values.append([int(value)])
            else:
                try:
                    eval_value = eval(value)
                    if isinstance(eval_value, (int, float, str)):
                        variable_values.append([eval_value])
                except:
                    variable_values.append([value])

        for param_set in parameter_pattern.findall(code):
            params = param_set.split(',')
            for param in params:
                param = param.strip()
                if '=' in param:
                    name, value = param.split('=')
                    name, value = name.strip(), value.strip()
                    try:
                        eval_value = eval(value)
                        if isinstance(eval_value, (int, float, str)):
                            variable_values.append([eval_value])
                    except:
                        variable_values.append([value])

        return {
            'variable_names': list(all_variable_names) + method_names + class_names,
            'variable_values': variable_values,
            'strings': strings,
            'comments': comments
        }

