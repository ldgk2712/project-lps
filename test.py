from sympy import symbols, simplify, solve, Symbol, S
from typing import Dict, Any

def auto_simplex(A, b, c, problem_type='min'):
    """
    Giải bài toán quy hoạch tuyến tính bằng phương pháp đơn hình
    
    Args:
        A: Ma trận hệ số ràng buộc
        b: Vector vế phải
        c: Vector hệ số hàm mục tiêu
        problem_type: 'min' hoặc 'max'
    """
    # Nếu là bài toán max, chuyển thành min bằng cách đổi dấu c
    original_type = problem_type
    if problem_type == 'max':
        c = [-coeff for coeff in c]
        print(f"Chuyển bài toán MAX thành MIN: c = {c}")
    
    m, n = len(A), len(A[0])
    x_syms = list(symbols(f'x1:{n+1}'))
    w_syms = list(symbols(f'w1:{m+1}'))

    # Khởi tạo bảng đơn hình
    step = 0
    exprs: Dict[str, Any] = {}
    current_exprs = {}
    for i in range(m):
        lhs = sum(A[i][j] * x_syms[j] for j in range(n))
        current_exprs[str(w_syms[i])] = simplify(b[i] - lhs)
    z_expr = simplify(-sum(c[j] * x_syms[j] for j in range(n)) + sum(0 * w for w in w_syms))
    current_exprs['z'] = z_expr
    exprs[f'Step {step}'] = current_exprs.copy()

    basic_vars = [str(w) for w in w_syms]
    non_basic_vars = [str(x) for x in x_syms]

    while True:
        z_expr = current_exprs['z']
        # Chọn biến vào: hệ số âm nhất
        entering = None
        max_coeff = S.Zero
        for x in non_basic_vars:
            coeff = z_expr.coeff(Symbol(x))
            if coeff < 0 and coeff < max_coeff:
                max_coeff = coeff
                entering = x

        if entering is None or max_coeff == 0:
            print("Đã đạt nghiệm tối ưu.")
            break

        # Chọn biến ra: tỷ số |b_i / a_ij| nhỏ nhất, b_i >= 0, a_ij < 0
        min_ratio = float('inf')
        leaving = None
        for w in basic_vars:
            expr = current_exprs[w]
            coeff = expr.coeff(Symbol(entering))
            if coeff < 0:
                const = expr.subs([(Symbol(x), 0) for x in non_basic_vars])
                if const >= 0:
                    ratio = abs(const / coeff)
                    if ratio < min_ratio:
                        min_ratio = ratio
                        leaving = w

        if leaving is None:
            break  # Không còn biến ra hợp lệ

        # Pivot
        step += 1
        pivot_eq = current_exprs[leaving]
        # Giải để tìm entering, giữ leaving trong biểu thức
        pivot_solutions = solve(pivot_eq - Symbol(leaving), Symbol(entering))
        if not pivot_solutions:
            print(f"Không thể pivot tại bước {step}. Kiểm tra biểu thức: {pivot_eq}")
            break
        pivot_expr = pivot_solutions[0]

        new_exprs = {}
        # Cập nhật các biểu thức
        for var, expr in current_exprs.items():
            if var == leaving:
                new_exprs[entering] = simplify(pivot_expr)
            else:
                # Thay entering vào expr, giữ leaving (w2)
                new_exprs[var] = simplify(expr.subs(Symbol(entering), pivot_expr))

        # Cập nhật biến
        basic_vars[basic_vars.index(leaving)] = entering
        non_basic_vars[non_basic_vars.index(entering)] = leaving
        if leaving not in non_basic_vars:
            non_basic_vars.append(leaving)

        current_exprs = new_exprs
        exprs[f'Step {step} ({entering} in, {leaving} out)'] = current_exprs.copy()

    # In kết quả
    for title, content in exprs.items():
        print(f"\n{title}")
        for var, expr in content.items():
            print(f"{var} = {expr}")

    # In nghiệm tối ưu
    final_step = list(exprs.keys())[-1]
    optimal_z_prime = exprs[final_step]['z']
    
    # Tìm các biến không cơ bản (non-basic variables)
    basic_vars_final = [var for var in exprs[final_step].keys() if var != 'z']
    all_vars = [f'x{i}' for i in range(1, n+1)] + [f'w{i}' for i in range(1, m+1)]
    non_basic_vars_final = [var for var in all_vars if var not in basic_vars_final]
    
    # Tính z' khi các biến không cơ bản = 0
    z_prime_value = optimal_z_prime.subs([(Symbol(var), 0) for var in non_basic_vars_final])
    z_optimal = z_prime_value
    
    # Nếu bài toán ban đầu là MAX, cần đổi dấu kết quả z
    if original_type == 'max':
        z_optimal = -z_optimal
    
    print(f"\nOptimal z ({original_type}): {z_optimal}")
    print("Optimal solution:")
    
    # In giá trị các biến x (biến quyết định ban đầu)
    for i in range(1, n+1):
        var_name = f'x{i}'
        if var_name in basic_vars_final:
            # Biến cơ bản: tính giá trị khi non-basic = 0
            var_expr = exprs[final_step][var_name]
            var_value = var_expr.subs([(Symbol(var), 0) for var in non_basic_vars_final])
            print(f"{var_name} = {var_value}")
        else:
            # Biến không cơ bản = 0
            print(f"{var_name} = 0")

    return exprs

# Ví dụ sử dụng:
print("=== Bài toán MIN ===")
A = [[-3, 1], [1, 2]]
b = [6, 4]
c = [-1, 4]
auto_simplex(A, b, c, 'min')

print("\n\n=== Bài toán MAX ===")
A = [[-3, 1], [1, 2]]
b = [6, 4]
c = [1, -4]  # Hệ số ban đầu cho bài toán MAX
auto_simplex(A, b, c, 'max')