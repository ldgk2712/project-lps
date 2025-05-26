from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, LpStatus, lpSum

def solve_simplex(A, b, c, constraint_types, objective_type, variable_types):
    # Tạo bài toán
    prob = LpProblem("Linear_Programming", LpMaximize if objective_type == 'Max' else LpMinimize)
    
    # Tạo biến
    vars = []
    for i in range(len(c)):
        if variable_types[i] == '>=0':
            vars.append(LpVariable(f'x{i+1}', lowBound=0))
        elif variable_types[i] == '<=0':
            vars.append(LpVariable(f'x{i+1}', upBound=0))
        else:  # URS
            vars.append(LpVariable(f'x{i+1}'))
    
    # Hàm mục tiêu
    prob += lpSum(c[i] * vars[i] for i in range(len(c)))
    
    # Ràng buộc
    for i in range(len(A)):
        expr = lpSum(A[i][j] * vars[j] for j in range(len(A[i])))
        if constraint_types[i] == '<=':
            prob += expr <= b[i]
        elif constraint_types[i] == '>=':
            prob += expr >= b[i]
        else:
            prob += expr == b[i]
    
    # Giải bài toán
    prob.solve()
    
    # Kết quả
    status = LpStatus[prob.status]
    solution = {f'x{i+1}': vars[i].varValue for i in range(len(vars)) if vars[i].varValue is not None}
    objective_value = prob.objective.value() if prob.objective.value() is not None else None
    
    result = {
        'status': 'Tối ưu (Optimal)' if status == 'Optimal' else 'Vô nghiệm (Infeasible)' if status == 'Infeasible' else 'Vô hạn (Unbounded)',
        'solution': solution,
        'objectiveValue': objective_value,
        'steps': []  # PuLP không cung cấp bước chi tiết, cần tự triển khai nếu muốn
    }
    
    return result