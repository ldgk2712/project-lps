from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, LpStatus, lpSum

def solve_simplex(A, b, c, constraint_types, objective_type, variable_types):
    # Kiểm tra hàm mục tiêu bằng 0
    if all(coef == 0 for coef in c):
        return {
            'status': 'Lỗi: Hàm mục tiêu bằng 0',
            'solution': {},
            'objectiveValue': None,
            'steps': [],
            'multiple_solutions': False
        }
    
    # Kiểm tra ràng buộc không xác định
    all_zero_constraints = all(all(coef == 0 for coef in row) and b[i] == 0 for i, row in enumerate(A))
    if all_zero_constraints:
        return {
            'status': 'Lỗi: Ràng buộc không xác định',
            'solution': {},
            'objectiveValue': None,
            'steps': [],
            'multiple_solutions': False
        }

    # Tạo bài toán chính
    prob = LpProblem("Linear_Programming", LpMaximize if objective_type == 'Max' else LpMinimize)
    
    # Tạo biến
    vars = []
    for i in range(len(c)):
        if variable_types[i] == '>=0':
            vars.append(LpVariable(f'x{i+1}', lowBound=0))
        elif variable_types[i] == '<=0':
            vars.append(LpVariable(f'x{i+1}', upBound=0))
        else:
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
    
    # Kiểm tra trạng thái
    if status not in ['Optimal', 'Infeasible', 'Unbounded']:
        return {
            'status': 'Lỗi: Bài toán không hợp lệ',
            'solution': {},
            'objectiveValue': None,
            'steps': [],
            'multiple_solutions': False
        }
    
    # Kiểm tra vô số nghiệm (cho n=2)
    multiple_solutions = False
    solutions = [(solution, objective_value)]
    
    if status == 'Optimal' and len(c) == 2:  # Chỉ kiểm tra cho bài toán 2 biến
        # Thử tìm nghiệm khác bằng cách thêm ràng buộc mới
        for i in range(len(vars)):
            prob_copy = LpProblem("Check_Multiple_Solutions", LpMaximize if objective_type == 'Max' else LpMinimize)
            vars_copy = []
            for j in range(len(c)):
                if variable_types[j] == '>=0':
                    vars_copy.append(LpVariable(f'x{j+1}', lowBound=0))
                elif variable_types[j] == '<=0':
                    vars_copy.append(LpVariable(f'x{j+1}', upBound=0))
                else:
                    vars_copy.append(LpVariable(f'x{j+1}'))
            prob_copy += lpSum(c[j] * vars_copy[j] for j in range(len(c)))
            for j in range(len(A)):
                expr = lpSum(A[j][k] * vars_copy[k] for k in range(len(A[j])))
                if constraint_types[j] == '<=':
                    prob_copy += expr <= b[j]
                elif constraint_types[j] == '>=':
                    prob_copy += expr >= b[j]
                else:
                    prob_copy += expr == b[j]
            # Thêm ràng buộc để tìm nghiệm khác
            prob_copy += vars_copy[i] != solution[f'x{i+1}']
            prob_copy.solve()
            if LpStatus[prob_copy.status] == 'Optimal' and abs(prob_copy.objective.value() - objective_value) < 1e-6:
                new_solution = {f'x{j+1}': vars_copy[j].varValue if vars_copy[j].varValue is not None else 0 for j in range(len(vars_copy))}
                solutions.append((new_solution, prob_copy.objective.value()))
                multiple_solutions = True
                break
    
    result = {
        'status': 'Tối ưu (Optimal)' if status == 'Optimal' else 'Vô nghiệm (Infeasible)' if status == 'Infeasible' else 'Vô hạn (Unbounded)',
        'solution': solutions[0][0],  # Trả về nghiệm đầu tiên
        'objectiveValue': objective_value,
        'steps': [],
        'multiple_solutions': multiple_solutions,
        'all_solutions': solutions if multiple_solutions else []
    }
    
    return result