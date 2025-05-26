from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, LpStatus, lpSum

def solve_simplex(A, b, c, constraint_types, objective_type, variable_types):
    # Kiểm tra hàm mục tiêu bằng 0
    if all(coef == 0 for coef in c):
        return {
            'status': 'Lỗi: Hàm mục tiêu bằng 0',
            'solution': {},
            'objectiveValue': None,
            'steps': []
        }
    
    # Kiểm tra ràng buộc không xác định (A=0, b=0)
    all_zero_constraints = all(all(coef == 0 for coef in row) and b[i] == 0 for i, row in enumerate(A))
    if all_zero_constraints:
        return {
            'status': 'Lỗi: Ràng buộc không xác định (tất cả hệ số và b bằng 0)',
            'solution': {},
            'objectiveValue': None,
            'steps': []
        }

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
    solution = {f'x{i+1}': vars[i].varValue if vars[i].varValue is not None else 0 for i in range(len(vars))}
    objective_value = prob.objective.value() if prob.objective.value() is not None else None
    
    # Kiểm tra trạng thái bài toán
    if status not in ['Optimal', 'Infeasible', 'Unbounded']:
        status = 'Lỗi: Bài toán không hợp lệ'
        objective_value = None
        solution = {}
    
    result = {
        'status': 'Tối ưu (Optimal)' if status == 'Optimal' else 'Vô nghiệm (Infeasible)' if status == 'Infeasible' else 'Vô hạn (Unbounded)' if status == 'Unbounded' else status,
        'solution': solution,
        'objectiveValue': objective_value,
        'steps': []
    }
    
    return result