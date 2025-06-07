import logging
from typing import List, Dict, Tuple, Any, Union, Optional
from sympy import simplify, solve, Symbol, S, sympify, Add, Mul, Number, Expr

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dung sai cho các phép so sánh số thực
SIMPLEX_TOLERANCE = 1e-9

def get_bland_key(var_obj: Union[str, Symbol]) -> Tuple[int, int, int, str]:
    """
    Tạo khóa để sắp xếp các biến theo quy tắc Bland, ưu tiên x, y, w, a và các hậu tố _p, _n.
    x, y: biến quyết định (y là biến thay thế cho x <= 0)
    w: biến bù
    a: biến giả
    x_p, x_n: thành phần dương và âm của biến URS
    """
    var_name_str = str(var_obj)
    type_priority = 99
    main_index = 9999
    sub_index = 0 # _p sẽ là 1, _n sẽ là 2

    if '_p' in var_name_str:
        sub_index = 1
        base_name = var_name_str[:-2]
    elif '_n' in var_name_str:
        sub_index = 2
        base_name = var_name_str[:-2]
    else:
        base_name = var_name_str

    if base_name.startswith('x') or base_name.startswith('y'):
        type_priority = 0
    elif base_name.startswith('w'):
        type_priority = 1
    elif base_name.startswith('a') or base_name == 'x0': # x0 là biến giả trong pha 1
        type_priority = 2
    
    try:
        numeric_part = ''.join(filter(str.isdigit, base_name))
        main_index = int(numeric_part) if numeric_part else (0 if len(base_name) == 1 else 9998)
    except ValueError:
        pass # Giữ giá trị mặc định nếu không có số

    return (type_priority, main_index, sub_index, var_name_str)


def format_expression_for_printing(expression: Any) -> str:
    """Định dạng biểu thức SymPy để in, làm tròn và sắp xếp các số hạng."""
    if not isinstance(expression, (Add, Mul, Symbol, Number, Expr)):
        try:
            expression = sympify(expression)
        except (SyntaxError, TypeError, AttributeError):
            return str(expression)

    expression = simplify(expression)

    if expression.is_Number:
        try:
            num_val = float(expression.evalf(chop=True))
            if abs(num_val) < SIMPLEX_TOLERANCE / 100: return "0.00"
            return f"{num_val:.2f}"
        except (TypeError, ValueError): return str(expression)

    const_term = expression.as_coeff_Add()[0]
    var_terms_expr = expression - const_term

    try:
        const_val = float(const_term.evalf(chop=True))
        const_str = "0.00" if abs(const_val) < SIMPLEX_TOLERANCE / 100 else f"{const_val:.2f}"
    except (TypeError, ValueError): const_str = str(const_term)

    if var_terms_expr == S.Zero: return const_str

    var_term_dict: Dict[Symbol, Any] = {}
    terms_to_process = var_terms_expr.as_ordered_terms() if isinstance(var_terms_expr, Add) else [var_terms_expr]

    for term in terms_to_process:
        coeff, var_part = term.as_coeff_Mul()
        var_sym: Optional[Symbol] = None
        # Đảm bảo var_part là một Symbol duy nhất
        if var_part.is_Symbol:
            var_sym = var_part
        # Xử lý trường hợp var_part là một tích (ví dụ: 2.0*x1)
        elif len(term.free_symbols) == 1:
            var_sym = list(term.free_symbols)[0]
            coeff = term.coeff(var_sym)
        else: # Bỏ qua các số hạng phức tạp không mong muốn
            continue
            
        if var_sym is not None: var_term_dict[var_sym] = var_term_dict.get(var_sym, S.Zero) + coeff

    sorted_var_symbols = sorted(var_term_dict.keys(), key=get_bland_key)
    
    var_str_list = []
    for var_sym in sorted_var_symbols:
        coeff = var_term_dict[var_sym]
        try:
            coeff_val = float(coeff.evalf(chop=True))
            if abs(coeff_val) < SIMPLEX_TOLERANCE / 100: continue
            
            sign = "+ " if coeff_val > 0 else "- "
            abs_coeff_val = abs(coeff_val)
            
            term_str_val_part = str(var_sym) if abs(abs_coeff_val - 1.0) < SIMPLEX_TOLERANCE else f"{abs_coeff_val:.2f}*{var_sym}"
            var_str_list.append((sign, term_str_val_part))
        except (TypeError, ValueError): # Nếu hệ số là biểu thức
            sign_str = "+ " if not coeff.is_negative else "- "
            abs_coeff = abs(coeff)
            term_str_val_part = str(var_sym) if abs_coeff == 1 else f"{abs_coeff}*{var_sym}"
            var_str_list.append((sign_str, term_str_val_part))

    if not var_str_list: return const_str
    
    result_str = const_str if const_str != "0.00" else ""
    for i, (sign, term_s) in enumerate(var_str_list):
        if result_str:
            result_str += f" {sign}{term_s}"
        else:
            result_str = term_s if sign == "+ " else f"-{term_s}"
            
    return result_str if result_str else "0.00"

def _get_ordered_tableau_for_history(tableau_dict: Dict[str, Expr], objective_var_name: str) -> List[Tuple[str, str]]:
    """Sắp xếp và định dạng bảng để lưu vào lịch sử các bước."""
    ordered_vars = []
    if objective_var_name in tableau_dict:
        ordered_vars.append(objective_var_name)
    
    other_keys_list = sorted(
        [k for k in tableau_dict.keys() if k != objective_var_name], 
        key=get_bland_key
    )
    ordered_vars.extend(other_keys_list)
    
    return [(var, format_expression_for_printing(tableau_dict[var])) for var in ordered_vars if var in tableau_dict]

def _get_current_solution_values(
    tableau: Dict[str, Expr], 
    basic_vars: List[str], 
    non_basic_vars: List[str],
    original_var_info_map: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Tính toán giá trị của các biến GỐC tại một bước nhất định để vẽ đồ thị."""
    subs_dict = {Symbol(nb): 0 for nb in non_basic_vars}
    
    # Tính giá trị của tất cả các biến trong bảng (biến tableau)
    tableau_vars_values = {nb: 0.0 for nb in non_basic_vars}
    for bv in basic_vars:
        val = 0.0
        if bv in tableau:
            try:
                # Thay các biến không cơ sở bằng 0
                val = float(tableau[bv].subs(subs_dict).evalf())
            except (TypeError, ValueError): 
                pass
        tableau_vars_values[bv] = val
        
    # Từ giá trị biến tableau, tính ngược lại giá trị biến gốc
    solution_values_numeric = {}
    for info in original_var_info_map:
        original_name = info['original_name']
        
        if info.get('is_standard', False):
            tableau_name = info['tableau_name']
            solution_values_numeric[original_name] = tableau_vars_values.get(tableau_name, 0.0)
        
        elif info.get('is_transformed_neg', False):
            tableau_name = info['tableau_name']
            solution_values_numeric[original_name] = -tableau_vars_values.get(tableau_name, 0.0)
            
        elif info.get('is_urs', False):
            pos_name = info['pos_name']
            neg_name = info['neg_name']
            pos_val = tableau_vars_values.get(pos_name, 0.0)
            neg_val = tableau_vars_values.get(neg_name, 0.0)
            solution_values_numeric[original_name] = pos_val - neg_val
            
    return solution_values_numeric

def _reconstruct_final_solution(
    final_tableau: Dict[str, Expr], 
    final_basic_vars: List[str], 
    final_non_basic_vars: List[str],
    original_var_info_map: List[Dict[str, Any]]
) -> Dict[str, str]:
    """Tính và định dạng giá trị nghiệm cuối cùng cho các biến gốc."""
    final_values_numeric = _get_current_solution_values(
        final_tableau, final_basic_vars, final_non_basic_vars, original_var_info_map
    )
    
    solution_formatted = {}
    for name, value in final_values_numeric.items():
        # Sửa lỗi hiển thị "-0.00"
        if abs(value) < SIMPLEX_TOLERANCE:
            solution_formatted[name] = "0.00"
        else:
            solution_formatted[name] = f"{value:.2f}"
            
    return solution_formatted

def _simplex_core_solver(
    initial_tableau: Dict[str, Expr], 
    initial_basic_vars: List[str], 
    initial_non_basic_vars: List[str],
    objective_var_name: str, 
    phase_name: str, 
    original_var_info_map: List[Dict[str, Any]]
) -> Tuple[str, Optional[float], Dict[str, Expr], Dict[str, Any], List[str], List[str]]:
    """Lõi giải Simplex, giải bài toán MIN, sử dụng quy tắc Bland."""
    tableau = {k: sympify(v) for k, v in initial_tableau.items()}
    basic_vars = initial_basic_vars[:]
    non_basic_vars = initial_non_basic_vars[:]
    
    steps_history: Dict[str, Any] = {}
    max_iterations, iteration_count = 100, 0
    
    # Ghi lại trạng thái ban đầu (sau khi đã xoay pivot cho pha 1 nếu cần)
    title_initial = f'{phase_name} - Bước {"1 (Sau xoay khởi đầu)" if phase_name == "Phase 1" else "0 (Bảng khởi tạo Pha 2)"}'
    coords_dict_initial = _get_current_solution_values(tableau, basic_vars, non_basic_vars, original_var_info_map)
    # Lấy tên biến gốc (x1, x2, ...) để đảm bảo thứ tự tọa độ
    original_var_names_sorted = sorted([info['original_name'] for info in original_var_info_map])
    steps_history[title_initial] = {
        'tableau': _get_ordered_tableau_for_history(tableau, objective_var_name),
        'coords': [coords_dict_initial.get(name, 0.0) for name in original_var_names_sorted],
    }
    
    current_step_num = 1 if phase_name == "Phase 1" else 0

    while iteration_count < max_iterations:
        iteration_count += 1
        obj_row_expr = tableau[objective_var_name]
        
        # 1. Chọn biến vào (Entering variable) theo quy tắc Bland
        entering_var_sym: Optional[Symbol] = None
        # Sắp xếp các ứng viên không cơ sở theo quy tắc Bland
        sorted_candidates = sorted([Symbol(s) for s in non_basic_vars], key=get_bland_key)

        for candidate_sym in sorted_candidates:
            coeff = obj_row_expr.coeff(candidate_sym)
            if coeff.is_Number and float(coeff.evalf()) < -SIMPLEX_TOLERANCE:
                entering_var_sym = candidate_sym
                break
        
        if entering_var_sym is None: # Đã tối ưu
            obj_val = float(obj_row_expr.as_coeff_Add()[0].evalf(chop=True))
            return 'Optimal', obj_val, tableau, steps_history, basic_vars, non_basic_vars

        # 2. Chọn biến ra (Leaving variable) theo quy tắc Bland
        potential_leaving = []
        for b_var_str in basic_vars:
            row_expr = tableau[b_var_str]
            # Hệ số của biến vào trong hàng ràng buộc
            coeff_in_row = row_expr.coeff(entering_var_sym)
            
            # Chỉ xét các hệ số pivot > 0 (tức là hệ số trong biểu thức < 0)
            if coeff_in_row.is_Number and float(coeff_in_row.evalf()) < -SIMPLEX_TOLERANCE:
                const_term = float(row_expr.as_coeff_Add()[0].evalf(chop=True))
                ratio = const_term / -float(coeff_in_row.evalf())
                if ratio >= -SIMPLEX_TOLERANCE: # Chấp nhận tỉ số không âm
                    potential_leaving.append((ratio, Symbol(b_var_str)))
        
        if not potential_leaving:
            return 'Unbounded', None, tableau, steps_history, basic_vars, non_basic_vars

        # Tìm tỉ số nhỏ nhất
        min_ratio_val = min(r for r, _ in potential_leaving)
        # Nếu có nhiều tỉ số bằng nhau, chọn biến ra theo quy tắc Bland
        tied_vars = [v for r, v in potential_leaving if abs(r - min_ratio_val) < SIMPLEX_TOLERANCE]
        leaving_var_sym = sorted(tied_vars, key=get_bland_key)[0]
        
        # 3. Xoay (Pivot)
        entering_var_str, leaving_var_str = str(entering_var_sym), str(leaving_var_sym)
        
        pivot_row_expr = tableau[leaving_var_str]
        coeff_entering_in_pivot = pivot_row_expr.coeff(entering_var_sym)
        
        # Biểu thức mới cho biến vào
        expr_for_entering_var = simplify((Symbol(leaving_var_str) - (pivot_row_expr - coeff_entering_in_pivot * entering_var_sym)) / coeff_entering_in_pivot)

        # Cập nhật bảng
        new_tableau = {var: expr.subs(entering_var_sym, expr_for_entering_var) for var, expr in tableau.items() if var != leaving_var_str}
        new_tableau[entering_var_str] = expr_for_entering_var
        tableau = {k: simplify(v) for k, v in new_tableau.items()}

        # Cập nhật danh sách biến cơ sở và không cơ sở
        basic_vars.remove(leaving_var_str)
        basic_vars.append(entering_var_str)
        non_basic_vars.remove(entering_var_str)
        non_basic_vars.append(leaving_var_str)
        
        # Ghi lại bước này
        current_step_num += 1
        title = f'{phase_name} - Bước {current_step_num} (Vào: {entering_var_str}, Ra: {leaving_var_str})'
        coords_dict = _get_current_solution_values(tableau, basic_vars, non_basic_vars, original_var_info_map)
        steps_history[title] = {
            'tableau': _get_ordered_tableau_for_history(tableau, objective_var_name),
            'coords': [coords_dict.get(name, 0.0) for name in original_var_names_sorted],
        }

    return "MaxIterations", None, tableau, steps_history, basic_vars, non_basic_vars

def simplex_two_phase(
    A_orig: List[List[float]], b_orig: List[float], c_orig: List[float],
    constraint_types_orig: List[str], variable_types_orig: List[str],
    objective_type_orig: str = 'max'
) -> Dict[str, Any]:
    
    num_original_vars = len(c_orig)
    m_orig = len(A_orig)
    is_max_problem = objective_type_orig.lower() == 'max'
    status_map_vn = {'Optimal': 'Tối ưu', 'Unbounded': 'Không giới nội', 'Infeasible': 'Vô nghiệm', 'Error': 'Lỗi', 'MaxIterations': 'Đạt giới hạn lặp'}

    # === BƯỚC 1: CHUẨN HÓA BIẾN ( xử lý <=0 và URS) ===
    tableau_decision_symbols: List[Symbol] = []
    c_tableau: List[float] = []
    original_var_info_map: List[Dict[str, Any]] = []

    for i in range(num_original_vars):
        original_name = f"x{i+1}"
        var_type = variable_types_orig[i].strip()
        info: Dict[str, Any] = {'original_name': original_name, 'type': var_type, 'original_idx': i}

        if var_type == '<=0':
            y_name = f"y{i+1}"
            info.update({'tableau_name': y_name, 'is_transformed_neg': True})
            tableau_decision_symbols.append(Symbol(y_name))
            c_tableau.append(-c_orig[i])
        elif var_type == 'URS':
            pos_name, neg_name = f"x{i+1}_p", f"x{i+1}_n"
            info.update({'pos_name': pos_name, 'neg_name': neg_name, 'is_urs': True})
            tableau_decision_symbols.extend([Symbol(pos_name), Symbol(neg_name)])
            c_tableau.extend([c_orig[i], -c_orig[i]])
        else: # '>=0'
            info.update({'tableau_name': original_name, 'is_standard': True})
            tableau_decision_symbols.append(Symbol(original_name))
            c_tableau.append(c_orig[i])
        original_var_info_map.append(info)

    A_tableau_cols = []
    for i in range(num_original_vars):
        var_type = variable_types_orig[i].strip()
        if var_type == '<=0':
            A_tableau_cols.append([-A_orig[j][i] for j in range(m_orig)])
        elif var_type == 'URS':
            A_tableau_cols.append([A_orig[j][i] for j in range(m_orig)])
            A_tableau_cols.append([-A_orig[j][i] for j in range(m_orig)])
        else: # '>=0'
            A_tableau_cols.append([A_orig[j][i] for j in range(m_orig)])
    
    # Transpose columns to get the final A_tableau matrix
    A_tableau = [[A_tableau_cols[c][r] for c in range(len(A_tableau_cols))] for r in range(m_orig)]

    # === BƯỚC 2: CHUYỂN RÀNG BUỘC VỀ DẠNG <= ===
    A_le, b_le = [], []
    for A_row_tableau, b_val, c_type in zip(A_tableau, b_orig, constraint_types_orig):
        if c_type == '>=':
            A_le.append([-c for c in A_row_tableau])
            b_le.append(-b_val)
        elif c_type == '=':
            A_le.append(A_row_tableau)
            b_le.append(b_val)
            A_le.append([-c for c in A_row_tableau])
            b_le.append(-b_val)
        else: # '<='
            A_le.append(A_row_tableau)
            b_le.append(b_val)

    # === BƯỚC 3: PHA 1 (PHASE 1) ===
    x0_sym = Symbol('x0')
    tableau_p1: Dict[str, Expr] = {}
    basic_vars_p1: List[str] = []
    non_basic_vars_p1 = [str(s) for s in tableau_decision_symbols] + [str(x0_sym)]
    
    # Tạo bảng Pha 1 ban đầu
    for i in range(len(b_le)):
        w_var = Symbol(f'w{i+1}')
        lhs_expr = sum(S(A_le[i][j]) * tableau_decision_symbols[j] for j in range(len(tableau_decision_symbols)))
        tableau_p1[str(w_var)] = S(b_le[i]) - lhs_expr + x0_sym # Thêm biến giả x0
        basic_vars_p1.append(str(w_var))

    p1_obj_name = 'W'
    tableau_p1[p1_obj_name] = x0_sym # Hàm mục tiêu W = x0 -> min W = min x0
    
    # Ghi lại bước 0 của pha 1
    steps_p1 = {}
    title_step0 = 'Phase 1 - Bước 0 (Bảng khởi tạo)'
    original_var_names_sorted = sorted([info['original_name'] for info in original_var_info_map])
    coords_step0 = _get_current_solution_values(tableau_p1, basic_vars_p1, non_basic_vars_p1, original_var_info_map)
    steps_p1[title_step0] = {
        'tableau': _get_ordered_tableau_for_history(tableau_p1, p1_obj_name),
        'coords': [coords_step0.get(name, 0.0) for name in original_var_names_sorted]
    }

    # Tìm biến ra để làm cho bảng khả thi (đưa x0 vào cơ sở)
    const_terms = {bv: float(tableau_p1[bv].subs({Symbol(s): 0 for s in non_basic_vars_p1}).evalf()) for bv in basic_vars_p1}
    # Biến ra là biến có hằng số âm nhất
    leaving_var_p0 = min(const_terms, key=const_terms.get)

    # Xoay pivot để đưa x0 (entering) vào cơ sở và leaving_var_p0 (leaving) ra
    entering_sym = x0_sym
    entering_str = str(entering_sym)
    leaving_str = leaving_var_p0

    pivot_row_expr = tableau_p1[leaving_str]
    coeff_entering = pivot_row_expr.coeff(entering_sym) # Đây là hệ số của x0, phải là 1

    # Biểu thức mới cho biến vào (x0)
    expr_for_entering = simplify((Symbol(leaving_str) - (pivot_row_expr - coeff_entering * entering_sym)) / coeff_entering)

    # Cập nhật bảng
    new_tableau_p1 = {var: simplify(expr.subs(entering_sym, expr_for_entering)) for var, expr in tableau_p1.items() if var != leaving_str}
    new_tableau_p1[entering_str] = expr_for_entering
    tableau_p1 = new_tableau_p1
        
    # Cập nhật danh sách biến
    basic_vars_p1.remove(leaving_str)
    basic_vars_p1.append(entering_str)
    non_basic_vars_p1.remove(entering_str)
    non_basic_vars_p1.append(leaving_str)
    
    # Giải bài toán Pha 1
    status_p1, w_min_val, final_tableau_p1, steps_p1_solver, final_basic_p1, final_non_basic_p1 = _simplex_core_solver(
        tableau_p1, basic_vars_p1, non_basic_vars_p1, p1_obj_name, "Phase 1", original_var_info_map
    )
    steps_p1.update(steps_p1_solver)

    # Kiểm tra kết quả Pha 1
    if status_p1 != 'Optimal' or (w_min_val is not None and w_min_val > SIMPLEX_TOLERANCE):
        z_infeasible = float('-inf') if is_max_problem else float('inf')
        error_msg = f"Pha 1 kết thúc với x0_min = {w_min_val:.4f} > 0. Bài toán gốc vô nghiệm."
        return {'status': status_map_vn['Infeasible'], 'z': str(z_infeasible), 'solution': {}, 'steps': steps_p1, 'error_message': error_msg}

    # === BƯỚC 4: PHA 2 (PHASE 2) ===
    # Chuẩn bị bảng cho Pha 2 từ kết quả Pha 1
    tableau_p2: Dict[str, Expr] = {}

    # Nếu x0 vẫn còn trong cơ sở với giá trị 0, ta cần xoay để loại nó ra
    if str(x0_sym) in final_basic_p1:
        # Tìm một biến không cơ sở khác x0 có hệ số khác 0 trong hàng x0 để xoay
        x0_row_expr = final_tableau_p1[str(x0_sym)]
        pivot_col_sym = None
        for nb_var_str in final_non_basic_p1:
            if abs(x0_row_expr.coeff(Symbol(nb_var_str))) > SIMPLEX_TOLERANCE:
                pivot_col_sym = Symbol(nb_var_str)
                break
        
        # Nếu tìm thấy, thực hiện xoay pivot
        if pivot_col_sym:
            pivot_col_str = str(pivot_col_sym)
            coeff = x0_row_expr.coeff(pivot_col_sym)
            expr_for_pivot_col = simplify((-(x0_row_expr - coeff * pivot_col_sym)) / coeff)
            
            for var, expr in final_tableau_p1.items():
                if var != p1_obj_name and var != str(x0_sym):
                    tableau_p2[var] = expr.subs(pivot_col_sym, expr_for_pivot_col)
            
            tableau_p2[pivot_col_str] = expr_for_pivot_col

            final_basic_p1.remove(str(x0_sym))
            final_basic_p1.append(pivot_col_str)
            final_non_basic_p1.remove(pivot_col_str)
            final_non_basic_p1.append(str(x0_sym))
        # Nếu không tìm thấy (trường hợp hiếm, ràng buộc dư thừa), chỉ cần loại bỏ hàng x0
        else:
             final_basic_p1.remove(str(x0_sym))

    # Loại bỏ biến giả x0 và hàm mục tiêu W
    for var, expr in final_tableau_p1.items():
        if var != p1_obj_name and var != str(x0_sym) and var not in tableau_p2:
            tableau_p2[var] = expr.subs(x0_sym, 0)

    basic_vars_p2 = [v for v in final_basic_p1 if v != str(x0_sym)]
    non_basic_vars_p2 = [v for v in final_non_basic_p1 if v != str(x0_sym)]
    
    # Xây dựng hàm mục tiêu cho Pha 2
    p2_obj_name = 'z'
    z_expr_orig = sum(S(c_tableau[i]) * tableau_decision_symbols[i] for i in range(len(c_tableau)))
    
    # Nếu là bài toán MAX, ta min(-Z)
    z_expr_to_solve = -z_expr_orig if is_max_problem else z_expr_orig
    
    # Thế các biến cơ sở vào hàm mục tiêu
    subs_dict_p2 = {Symbol(bv): tableau_p2[bv] for bv in basic_vars_p2 if Symbol(bv) in z_expr_to_solve.free_symbols and bv in tableau_p2}
    tableau_p2[p2_obj_name] = simplify(z_expr_to_solve.subs(subs_dict_p2))

    # Giải bài toán Pha 2
    status_p2, z_min_value, final_tableau_p2, steps_p2, final_basic_p2, final_non_basic_p2 = _simplex_core_solver(
        tableau_p2, basic_vars_p2, non_basic_vars_p2, p2_obj_name, "Phase 2", original_var_info_map
    )
    
    # === BƯỚC 5: XỬ LÝ KẾT QUẢ ===
    combined_steps = steps_p1.copy()
    combined_steps.update(steps_p2)
    
    final_z = "N/A"
    if status_p2 == 'Optimal' and z_min_value is not None:
        final_z = -z_min_value if is_max_problem else z_min_value
    elif status_p2 == 'Unbounded':
        final_z = float('inf') if is_max_problem else float('-inf')

    solution_final = {}
    if status_p2 == 'Optimal':
        solution_final = _reconstruct_final_solution(final_tableau_p2, final_basic_p2, final_non_basic_p2, original_var_info_map)

    return {
        'status': status_map_vn.get(status_p2, status_p2), 
        'z': f"{final_z:.2f}" if isinstance(final_z, (float, int)) else str(final_z),
        'solution': solution_final, 
        'steps': combined_steps, 
        'error_message': None
    }
