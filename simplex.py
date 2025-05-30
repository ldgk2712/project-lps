from sympy import symbols, simplify, solve, Symbol, S
from typing import List, Dict, Tuple, Any, Union

"""simplex_patch.py – Fix objective-type handling (MAX/MIN)

* Convert **MAX → MIN** by negating c when objective_type == 'max'.
* Keep c as‑is for MIN.
* Flip sign of z* back to MAX when returning.
* Retains preprocessing of ≥ constraints and step-by-step printing.
"""

# ----------------------------- helpers --------------------------------------

def _print_tableau(title: str, exprs: Dict[str, Any]) -> None:
    ordered = sorted(k for k in exprs if k != 'z') + (['z'] if 'z' in exprs else [])
    print(f"\n{title}")
    for var in ordered:
        print(f"  {var} = {exprs[var]}")


def _preprocess(A: List[List[float]], b: List[float], constraint_types: List[str]) -> Tuple[List[List[float]], List[float]]:
    """Convert every row to ≤ form by flipping ≥ rows."""
    return A, b

# --------------------------- core simplex -----------------------------------

def _simplex_min(A: List[List[float]], b: List[float], c: List[float], constraint_types: List[str], variable_types: List[str], objective_type: str) -> Tuple[str, Union[float, None], Dict[str, float], Dict[str, Any]]:
    m, n = len(A), len(A[0])
    x_syms = list(symbols(f"x1:{n+1}"))
    w_syms = list(symbols(f"w1:{m+1}"))

    # Chuẩn hóa ràng buộc = và biến x_i <= 0
    A_standard = [row[:] for row in A]
    b_standard = b[:]
    c_standard = c[:]
    num_vars = n
    all_vars = [f'x{i+1}' for i in range(n)]
    num_slack_vars = 0
    
    # Xử lý biến x_i <= 0: thay x_i bằng y_i = -x_i
    negative_vars = [i for i in range(n) if variable_types[i].strip() == '<=0']
    for i in negative_vars:
        # Thay cột i trong A_standard: a_ij = -a_ij (vì x_i = -y_i)
        for j in range(m):
            A_standard[j][i] = -A_standard[j][i]
        # Thay c_i trong c_standard: c_i = -c_i
        c_standard[i] = -c_standard[i]
    
    # Đếm số ràng buộc = để biết cần thêm bao nhiêu biến phụ
    num_equals = sum(1 for t in constraint_types if t.strip() == '=')
    
    # Mở rộng A_standard cho tất cả biến phụ trước
    for _ in range(num_equals):
        for row in A_standard:
            row.append(0)
    
    # Xử lý các ràng buộc =
    for i in range(m):
        if constraint_types[i].strip() == '=':
            current_slack_col = n + num_slack_vars
            A_standard[i][current_slack_col] = -1  # Hệ số của s_i là -1
            num_slack_vars += 1
            num_vars += 1  # Tăng số lượng biến
            c_standard.append(0)  # Hệ số của s_i trong hàm mục tiêu là 0
            all_vars.append(f's{num_slack_vars}')

    # Cập nhật x_syms với số lượng biến mới
    x_syms = list(symbols(f"x1:{num_vars+1}"))
    
    step = 0    
    steps: Dict[str, Dict[str, Any]] = {}
    cur: Dict[str, Any] = {}

    # Step 0: Khởi tạo bảng đơn hình, xử lý >= bằng cách chuyển vế và đổi dấu
    for i in range(m):
        if constraint_types[i].strip() == '>=':
            cur[str(w_syms[i])] = simplify(-b[i] + sum(A_standard[i][j] * x_syms[j] for j in range(num_vars)))
        else:  # <= or equality
            cur[str(w_syms[i])] = simplify(b[i] - sum(A_standard[i][j] * x_syms[j] for j in range(num_vars)))
    cur['z'] = simplify(sum(c_standard[j] * x_syms[j] for j in range(num_vars)))
    steps['Step 0'] = cur.copy()
    _print_tableau('Step 0', cur)

    basic = [str(w) for w in w_syms]
    non_basic = [str(x) for x in x_syms]

    # Kiểm tra vô nghiệm trước khi vào vòng lặp
    all_constraints_negative = True
    for i in range(m):
        expr = cur[f'w{i+1}']
        vars_to_sub = [(Symbol(var), 0) for var in non_basic]
        const = expr.subs(vars_to_sub)
        const = simplify(const)
        if not const.is_number:
            print(f"Lỗi: const không phải là số sau khi thay thế: {const}")
            continue
        const_value = float(const)
        if const_value >= 0:
            all_constraints_negative = False
            break
        can_become_feasible = False
        for j in range(num_vars):
            coeff = expr.coeff(Symbol(f'x{j+1}'))
            if coeff > 0:
                can_become_feasible = True
                break
        if can_become_feasible:
            all_constraints_negative = False
            break
    if all_constraints_negative:
        print("Bài toán vô nghiệm.(Miền xác nhận rỗng)")
        z_value = float('inf') if objective_type.strip().lower().startswith('max') else float('-inf')
        return 'Infeasible', z_value, {}, steps
    
    # Kiểm tra không giới nội trước khi vào vòng lặp
    for j in range(n):
        if c[j] < 0:  # Chỉ kiểm tra với MIN (đã chuẩn hóa)
            all_coeffs_nonpositive = True
            for i in range(m):
                expr = cur[f'w{i+1}']
                coeff = expr.coeff(Symbol(f'x{j+1}'))
                if coeff < 0:
                    all_coeffs_nonpositive = False
                    break
            if all_coeffs_nonpositive:
                print("Bài toán không giới nội.")
                z_value = float('-inf') if objective_type.strip().lower().startswith('max') else float('inf')
                return 'Unbounded', z_value, {}, steps
    
    while True:
        z_expr = cur['z']
        entering, most_neg = None, S.Zero
        for v in non_basic:
            coeff = z_expr.coeff(Symbol(v))
            if coeff < most_neg:
                most_neg = coeff
                entering = v
        if entering is None:
            status = 'Optimal'
            break

        leaving, min_ratio = None, float('inf')
        for w in basic:
            expr = cur[w]
            coeff = expr.coeff(Symbol(entering))
            if coeff < 0:
                has_negative_coeff = True
                vars_to_sub = [(Symbol(var), 0) for var in non_basic]
                const = expr.subs(vars_to_sub)
                const = simplify(const)
                if not const.is_number:
                    print(f"Lỗi: const không phải là số sau khi thay thế: {const}")
                    continue
                const_value = float(const)
                if const_value >= 0:
                    ratio = const_value / -coeff
                    if ratio < min_ratio:
                        min_ratio, leaving = ratio, w

        if has_negative_coeff and leaving is None:
            title = f"Step {step+1} (enter {entering}, no leaving — Unbounded)"
            steps[title] = cur.copy()
            _print_tableau(title, cur)
            z_value = float('-inf') if objective_type.strip().lower().startswith('max') else float('inf')
            return 'Unbounded', z_value, {}, steps
        elif leaving is None:
            title = f"Step {step+1} (enter {entering}, no leaving — Infeasible)"
            steps[title] = cur.copy()
            _print_tableau(title, cur)
            z_value = float('inf') if objective_type.strip().lower().startswith('max') else float('-inf')
            return 'Infeasible', z_value, {}, steps

        step += 1
        pivot_expr = solve(cur[leaving] - Symbol(leaving), Symbol(entering))[0]
        
        new_cur = {}
        for var, expr in cur.items():
            if var == leaving:
                # old leaving trở thành non‑basic, skip it for now
                continue
            else:
                new_cur[var] = simplify(expr.subs(Symbol(entering), pivot_expr))
        new_cur[entering] = simplify(pivot_expr)

         # cập nhật tập biến
        basic[basic.index(leaving)] = entering
        non_basic[non_basic.index(entering)] = leaving
        cur = new_cur

        title = f"Step {step} ({entering} in, {leaving} out)"
        steps[title] = cur.copy()
        _print_tableau(title, cur)
        
    all_vars = [str(x) for x in x_syms] + [str(w) for w in w_syms]
    subs0 = {Symbol(v): 0 for v in all_vars if v in non_basic}
    z_star = float(cur['z'].subs(subs0))
    opt_vals = {v: (float(cur[v].subs(subs0)) if v in cur else 0.0) for v in all_vars}

    # Chuyển kết quả từ y_i về x_i cho các biến x_i <= 0
    for i in negative_vars:
        var_name = f'x{i+1}'
        if var_name in opt_vals:
            opt_vals[var_name] = -opt_vals[var_name]  # x_i = -y_i
    
    # Tính lại z theo x_i
    if status == 'Optimal':
        z_star = sum(c[i] * opt_vals[f'x{i+1}'] for i in range(n))
    
    return status, z_star, opt_vals, steps

# --------------------------- public wrapper ---------------------------------

def auto_simplex(
    A: List[List[float]],
    b: List[float],
    c: List[float],
    constraint_types: List[str],
    objective_type: str = 'max',
    variable_types: List[str] | None = None,
) -> Dict[str, Any]:
    # Nếu không có variable_types, giả định tất cả biến >= 0
    if variable_types is None:
        variable_types = ['>=0'] * len(c)
    
    A_std, b_std = _preprocess(A, b, constraint_types)

    # MAX → MIN by negating c
    if objective_type and objective_type.strip().lower().startswith('max'):
        c_eff = [-ci for ci in c]
        flip_back = True
    else:
        c_eff = c[:]
        flip_back = False

    status, z_star, opt_vals, steps = _simplex_min(A_std, b_std, c_eff, constraint_types, variable_types, objective_type)

    # flip sign for MAX objective value
    if flip_back and z_star is not None:
        z_star = -z_star

    return {
        'status': {'Optimal': 'Tối ưu (Optimal)', 'Unbounded': 'Không giới nội (Unbounded)'}[status],
        'z': z_star,
        'solution': {k: v for k, v in opt_vals.items() if k.startswith('x')},
        'steps': steps,
    }

# ------------------------------- tests --------------------------------------
if __name__ == '__main__':
    # Feasible MIN
    c = [6, 8, 5, 9]
    A = [[2, 1, 1, 3],
        [1, 3, 1, 2]]
    b = [5, 3]
    cons1 = ['<=', '<=']
    print('\nMIN example:')
    print(auto_simplex(A, b, c, cons1, 'Min'))

    # Same LP as MAX
    print('\nMAX example:')
    print(auto_simplex(A, b, c, cons1, 'Max'))