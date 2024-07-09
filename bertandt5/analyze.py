
import re
def analyze_python_code(code):

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

    # Handle values for default parameters and assigned variables
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

    # Extract default parameter values
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
    # Patterns for variable, method, class, string, and comment
    variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s+[a-zA-Z_]\w*\s*=')
    parameter_pattern = re.compile(r'\b[a-zA-Z_]\w*\s+\w+\s*\((.*?)\)')
    string_pattern = re.compile(r'(\"[^\"]*\")')
    comment_pattern = re.compile(r'(//.*|/\*[\s\S]*?\*/)')
    method_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s+[a-zA-Z_]\w*\s*\(')
    class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*')

    # Extract variables, parameters, strings, comments, method names, and class names
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

    # Handle values for assigned variables and parameters
    value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*?);')
    variable_values = []

    # Extract variable values from assignments
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

    # Extract default parameter values from method definitions
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
    # Patterns for variable, function, class, string, and comment
    variable_pattern = re.compile(r'\b(let|var|const)\s+([a-zA-Z_]\w*)\s*=')
    parameter_pattern = re.compile(r'function\s+\w+\s*\((.*?)\)|\((.*?)\)\s*=>')
    string_pattern = re.compile(r'(\"[^\"]*\"|\'[^\']*\')')
    comment_pattern = re.compile(r'(//.*|/\*[\s\S]*?\*/)')
    function_pattern = re.compile(r'function\s+([a-zA-Z_]\w*)\s*\(')
    class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*')

    # Extract variables, parameters, strings, comments, function names, and class names
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

    # Handle values for assigned variables and parameters
    value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*?);')
    variable_values = []

    # Extract variable values from assignments
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

    # Extract default parameter values from function definitions
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
    # Patterns for variable, method, class, string, and comment
    variable_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*=')
    instance_variable_pattern = re.compile(r'@\w+')
    parameter_pattern = re.compile(r'def\s+\w+\s*\((.*?)\)')
    string_pattern = re.compile(r'(\'[^\']*\'|\"[^\"]*\")')
    comment_pattern = re.compile(r'#.*')
    method_pattern = re.compile(r'def\s+([a-zA-Z_]\w*)\s*')
    class_pattern = re.compile(r'class\s+([a-zA-Z_]\w*)\s*')

    # Extract variables, instance variables, parameters, strings, comments, method names, and class names
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

    # Handle values for assigned variables and parameters
    value_pattern = re.compile(r'([a-zA-Z_]\w*)\s*=\s*(.*?)(?:;|\n)')
    variable_values = []

    # Extract variable values from assignments
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

    # Extract default parameter values from method definitions
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

