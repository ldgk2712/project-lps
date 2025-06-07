import logging
from typing import List, Dict, Tuple, Any, Union, Optional
from sympy import simplify, Symbol, S, sympify, Add, Mul, Number, Expr

# Cấu hình logging để gỡ lỗi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dung sai cho các phép so sánh số thực để xử lý sai số tính toán
SIMPLEX_TOLERANCE = 1e-9

def get_bland_key(var_obj: Union[str, Symbol]) -> Tuple[int, int, int, str]:
    """
    Tạo khóa để sắp xếp các biến theo quy tắc Bland, chống xoay vòng (cycling).
    Ưu tiên sắp xếp: biến quyết định (x, y) -> biến bù (w) -> biến giả (a, x0).
    Đối với biến URS (x_p, x_n), _p sẽ được ưu tiên hơn _n.
    """
    var_name_str = str(var_obj)
    type_priority = 99
    main_index = 9999
    sub_index = 0  # _p sẽ là 1, _n sẽ là 2

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
    elif base_name.startswith('a') or base_name == 'x0':
        type_priority = 2
    
    try:
        numeric_part = ''.join(filter(str.isdigit, base_name))
        main_index = int(numeric_part) if numeric_part else (0 if len(base_name) == 1 else 9998)
    except ValueError:
        pass

    return (type_priority, main_index, sub_index, var_name_str)


def format_expression_for_printing(expression: Any) -> str:
    """
    Định dạng biểu thức SymPy để in ra cho đẹp, làm tròn số và sắp xếp các số hạng.
    Hữu ích cho việc hiển thị các bước của bảng đơn hình.
    """
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
        if var_part.is_Symbol:
            var_sym = var_part
        elif len(term.free_symbols) == 1:
            var_sym = list(term.free_symbols)[0]
            coeff = term.coeff(var_sym)
        else:
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
        except (TypeError, ValueError):
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

def _reconstruct_final_solution(
    final_tableau: Dict[str, Expr], 
    final_basic_vars: List[str], 
    final_non_basic_vars: List[str],
    original_var_info_map: List[Dict[str, Any]],
    status: str,
    parametric_vars: List[str]
) -> Dict[str, Any]:
    """
    Từ bảng cuối cùng, tính toán và định dạng giá trị nghiệm cho các biến GỐC.
    Xử lý trường hợp nghiệm duy nhất và vô số nghiệm (tham số).
    Định dạng đầu ra theo yêu cầu giao diện người dùng.
    """
    if status == 'MultipleOptimal' and parametric_vars:
        # --- 1. Tính biểu thức nghiệm ---
        params_sym = {Symbol(p) for p in parametric_vars}
        subs_dict = {Symbol(nb): 0 for nb in final_non_basic_vars if Symbol(nb) not in params_sym}
        solution_expressions = {}
        for info in original_var_info_map:
            original_name = info['original_name']
            expr = S.Zero
            if info.get('is_standard', False):
                t_name = info['tableau_name']
                if t_name in final_basic_vars: expr = final_tableau[t_name]
                elif Symbol(t_name) in params_sym: expr = Symbol(t_name)
            elif info.get('is_transformed_neg', False):
                t_name = info['tableau_name']
                if t_name in final_basic_vars: expr = -final_tableau[t_name]
                elif Symbol(t_name) in params_sym: expr = -Symbol(t_name)
            elif info.get('is_urs', False):
                pos_name, neg_name = info['pos_name'], info['neg_name']
                pos_expr = final_tableau[pos_name] if pos_name in final_basic_vars else (Symbol(pos_name) if pos_name in parametric_vars else S.Zero)
                neg_expr = final_tableau[neg_name] if neg_name in final_basic_vars else (Symbol(neg_name) if neg_name in parametric_vars else S.Zero)
                expr = pos_expr - neg_expr
            solution_expressions[original_name] = simplify(expr.subs(subs_dict))

        # --- 2. Tìm và ánh xạ điều kiện cho từng tham số ---
        param_conditions_map = {}
        vars_to_check_non_negativity = final_basic_vars + parametric_vars
        
        for p in parametric_vars:
            param_conditions_map[p] = {'lower': S.Zero, 'upper': S.Infinity}

        for var_str in vars_to_check_non_negativity:
            if var_str in final_basic_vars:
                expr = final_tableau[var_str].subs(subs_dict)
            else:
                expr = Symbol(var_str)
            expr_params = [p for p in parametric_vars if Symbol(p) in expr.free_symbols]

            if len(expr_params) == 1:
                param_str = expr_params[0]
                param_sym = Symbol(param_str)
                const_term = expr.subs(param_sym, 0)
                coeff = expr.coeff(param_sym)
                if abs(float(coeff.evalf())) > SIMPLEX_TOLERANCE:
                    bound = simplify(-const_term / coeff)
                    if float(coeff.evalf()) > 0:
                        current_lower = param_conditions_map[param_str]['lower']
                        if bound.is_Number and (current_lower is S.NegativeInfinity or bound > current_lower):
                            param_conditions_map[param_str]['lower'] = bound
                    else:
                        current_upper = param_conditions_map[param_str]['upper']
                        if bound.is_Number and (current_upper is S.Infinity or bound < current_upper):
                            param_conditions_map[param_str]['upper'] = bound
        
        # --- 3. Định dạng chuỗi điều kiện cho từng tham số ---
        param_condition_strings = {}
        for p_str, bounds in param_conditions_map.items():
            lower_bound_expr = simplify(bounds['lower'])
            upper_bound_expr = simplify(bounds['upper'])
            lower_val = float(lower_bound_expr.evalf()) if lower_bound_expr.is_Number else -float('inf')
            upper_val = float(upper_bound_expr.evalf()) if upper_bound_expr.is_Number else float('inf')

            if lower_val > upper_val + SIMPLEX_TOLERANCE:
                param_condition_strings[p_str] = f"Mâu thuẫn"
                continue
            
            lower_str = format_expression_for_printing(lower_bound_expr)
            upper_str = format_expression_for_printing(upper_bound_expr)
            
            if lower_str == "0.00" and upper_val == float('inf'):
                param_condition_strings[p_str] = f"{p_str} >= 0"
            elif upper_val != float('inf'):
                param_condition_strings[p_str] = f"{lower_str} <= {p_str} <= {upper_str}"
            else:
                param_condition_strings[p_str] = f"{p_str} >= {lower_str}"
        
        # --- 4. Cấu trúc lại dữ liệu trả về theo giao diện yêu cầu ---
        solution_list = []
        for var_name, expr in sorted(solution_expressions.items()):
            expression_str = format_expression_for_printing(expr)
            note = ""
            
            params_in_expr = [p for p in parametric_vars if Symbol(p) in expr.free_symbols]

            if params_in_expr:
                specific_conditions = sorted([param_condition_strings.get(p, f"{p} >= 0") for p in params_in_expr])
                conditions_str = ", ".join(specific_conditions)

                is_base_parameter = expr.is_Symbol and str(expr) in parametric_vars
                if is_base_parameter:
                    note = f"(tham số, {conditions_str})"
                else:
                    note = f"(phụ thuộc tham số, {conditions_str})"
            
            solution_list.append({
                "variable": var_name,
                "expression": expression_str,
                "note": note,
            })

        return {
            "type": "parametric", 
            "solution": solution_list,
        }

    else: # 'Optimal' - Nghiệm duy nhất
        subs_dict = {Symbol(nb): 0 for nb in final_non_basic_vars}
        tableau_vars_values = {nb: 0.0 for nb in final_non_basic_vars}
        for bv in final_basic_vars:
            val = 0.0
            if bv in final_tableau:
                try: val = float(final_tableau[bv].subs(subs_dict).evalf())
                except (TypeError, ValueError): pass
            tableau_vars_values[bv] = val
        
        solution_values_numeric = {}
        for info in original_var_info_map:
            original_name = info['original_name']
            if info.get('is_standard', False):
                solution_values_numeric[original_name] = tableau_vars_values.get(info['tableau_name'], 0.0)
            elif info.get('is_transformed_neg', False):
                solution_values_numeric[original_name] = -tableau_vars_values.get(info['tableau_name'], 0.0)
            elif info.get('is_urs', False):
                pos_val = tableau_vars_values.get(info['pos_name'], 0.0)
                neg_val = tableau_vars_values.get(info['neg_name'], 0.0)
                solution_values_numeric[original_name] = pos_val - neg_val
        
        solution_formatted = {name: f"{value:.2f}" if abs(value) > SIMPLEX_TOLERANCE else "0.00" for name, value in solution_values_numeric.items()}
            
        solution_list = []
        for name, value in sorted(solution_formatted.items()):
            solution_list.append({
                "variable": name,
                "expression": value,
                "note": ""
            })
        return {"type": "point", "solution": solution_list}

def _simplex_core_solver(
    initial_tableau: Dict[str, Expr], 
    initial_basic_vars: List[str], 
    initial_non_basic_vars: List[str],
    objective_var_name: str, 
    phase_name: str,
    start_step: int = 1
) -> Tuple[str, Optional[float], Dict[str, Expr], Dict[str, Any], List[str], List[str], Dict[str, Any]]:
    """
    Lõi giải quyết của thuật toán đơn hình. Trả về thông tin bổ sung (ví dụ: biến tham số).
    """
    tableau = {k: sympify(v) for k, v in initial_tableau.items()}
    basic_vars = initial_basic_vars[:]
    non_basic_vars = initial_non_basic_vars[:]
    
    steps_history: Dict[str, Any] = {}
    max_iterations, iteration_count = 100, 0
    current_step_num = start_step

    while iteration_count < max_iterations:
        iteration_count += 1
        obj_row_expr = tableau[objective_var_name]
        
        # === 1. CHỌN BIẾN VÀO (ENTERING VARIABLE) ===
        entering_var_sym: Optional[Symbol] = None
        sorted_candidates = sorted([Symbol(s) for s in non_basic_vars], key=get_bland_key)

        for candidate_sym in sorted_candidates:
            coeff = obj_row_expr.coeff(candidate_sym)
            if coeff.is_Number and float(coeff.evalf()) < -SIMPLEX_TOLERANCE:
                entering_var_sym = candidate_sym
                break
        
        if entering_var_sym is None:
            # TỐI ƯU hoặc VÔ SỐ NGHIỆM TỐI ƯU
            obj_val = float(obj_row_expr.as_coeff_Add()[0].evalf(chop=True))
            
            parametric_vars = []
            for nb_var_str in non_basic_vars:
                if 'x0' in nb_var_str or str(nb_var_str).startswith('a'): continue # Bỏ qua biến giả
                coeff = obj_row_expr.coeff(Symbol(nb_var_str))
                if coeff.is_Number and abs(float(coeff.evalf())) < SIMPLEX_TOLERANCE:
                    parametric_vars.append(nb_var_str)

            status = 'MultipleOptimal' if parametric_vars else 'Optimal'
            return status, obj_val, tableau, steps_history, basic_vars, non_basic_vars, {'parametric_vars': parametric_vars}

        # === 2. CHỌN BIẾN RA (LEAVING VARIABLE) ===
        potential_leaving = []
        for b_var_str in basic_vars:
            row_expr = tableau[b_var_str]
            coeff_in_row = row_expr.coeff(entering_var_sym)
            
            if coeff_in_row.is_Number and float(coeff_in_row.evalf()) < -SIMPLEX_TOLERANCE:
                const_term = float(row_expr.as_coeff_Add()[0].evalf(chop=True))
                ratio = const_term / -float(coeff_in_row.evalf())
                if ratio >= -SIMPLEX_TOLERANCE:
                    potential_leaving.append((ratio, Symbol(b_var_str)))
        
        if not potential_leaving:
            return 'Unbounded', None, tableau, steps_history, basic_vars, non_basic_vars, {}

        min_ratio_val = min(r for r, _ in potential_leaving)
        tied_vars = [v for r, v in potential_leaving if abs(r - min_ratio_val) < SIMPLEX_TOLERANCE]
        leaving_var_sym = sorted(tied_vars, key=get_bland_key)[0]
        
        # === 3. XOAY BẢNG (PIVOT) ===
        entering_var_str, leaving_var_str = str(entering_var_sym), str(leaving_var_sym)
        
        pivot_row_expr = tableau[leaving_var_str]
        coeff_entering_in_pivot = pivot_row_expr.coeff(entering_var_sym)
        expr_for_entering_var = simplify((Symbol(leaving_var_str) - (pivot_row_expr - coeff_entering_in_pivot * entering_var_sym)) / coeff_entering_in_pivot)

        new_tableau = {var: expr.subs(entering_var_sym, expr_for_entering_var) for var, expr in tableau.items() if var != leaving_var_str}
        new_tableau[entering_var_str] = expr_for_entering_var
        tableau = {k: simplify(v) for k, v in new_tableau.items()}

        basic_vars.remove(leaving_var_str)
        basic_vars.append(entering_var_str)
        non_basic_vars.remove(entering_var_str)
        non_basic_vars.append(leaving_var_str)
        
        title = f'{phase_name} - Bước {current_step_num} (Vào: {entering_var_str}, Ra: {leaving_var_str})'
        steps_history[title] = {'tableau': _get_ordered_tableau_for_history(tableau, objective_var_name)}
        current_step_num += 1

    return "MaxIterations", None, tableau, steps_history, basic_vars, non_basic_vars, {}

def simplex_two_phase(
    A_orig: List[List[float]], b_orig: List[float], c_orig: List[float],
    constraint_types_orig: List[str], variable_types_orig: List[str],
    objective_type_orig: str = 'max'
) -> Dict[str, Any]:
    """
    Giải bài toán Quy hoạch tuyến tính bằng phương pháp Hai Pha (Two-Phase Simplex).
    """
    
    num_original_vars = len(c_orig)
    m_orig = len(A_orig)
    is_max_problem = objective_type_orig.lower() == 'max'
    status_map_vn = {
        'Optimal': 'Tối ưu', 
        'MultipleOptimal': 'Vô số nghiệm tối ưu',
        'Unbounded': 'Không giới nội', 
        'Infeasible': 'Vô nghiệm', 
        'Error': 'Lỗi', 
        'MaxIterations': 'Đạt giới hạn lặp'
    }

    # ===================================================================================
    # BƯỚC 1: CHUẨN HÓA BIẾN
    # ===================================================================================
    tableau_decision_symbols: List[Symbol] = []
    c_std: List[float] = []
    original_var_info_map: List[Dict[str, Any]] = []

    for i in range(num_original_vars):
        original_name = f"x{i+1}"
        var_type = variable_types_orig[i].strip().lower()
        info: Dict[str, Any] = {'original_name': original_name, 'type': var_type, 'original_idx': i}

        if var_type == '<=0':
            y_name = f"y{i+1}"
            info.update({'tableau_name': y_name, 'is_transformed_neg': True})
            tableau_decision_symbols.append(Symbol(y_name))
            c_std.append(-c_orig[i])
        elif var_type == 'urs':
            pos_name, neg_name = f"x{i+1}_p", f"x{i+1}_n"
            info.update({'pos_name': pos_name, 'neg_name': neg_name, 'is_urs': True})
            tableau_decision_symbols.extend([Symbol(pos_name), Symbol(neg_name)])
            c_std.extend([c_orig[i], -c_orig[i]])
        else: # '>=0'
            info.update({'tableau_name': original_name, 'is_standard': True})
            tableau_decision_symbols.append(Symbol(original_name))
            c_std.append(c_orig[i])
        original_var_info_map.append(info)

    A_cols_std = []
    for i in range(num_original_vars):
        var_type = variable_types_orig[i].strip().lower()
        if var_type == '<=0':
            A_cols_std.append([-A_orig[j][i] for j in range(m_orig)])
        elif var_type == 'urs':
            A_cols_std.append([A_orig[j][i] for j in range(m_orig)])
            A_cols_std.append([-A_orig[j][i] for j in range(m_orig)])
        else: # '>=0'
            A_cols_std.append([A_orig[j][i] for j in range(m_orig)])
    A_std = [[A_cols_std[c][r] for c in range(len(A_cols_std))] for r in range(m_orig)]

    # ===================================================================================
    # BƯỚC 2: CHUẨN HÓA RÀNG BUỘC
    # ===================================================================================
    A_le, b_le = [], []
    for A_row, b_val, c_type in zip(A_std, b_orig, constraint_types_orig):
        c_type = c_type.strip()
        if c_type == '>=':
            A_le.append([-c for c in A_row])
            b_le.append(-b_val)
        elif c_type == '=':
            A_le.append(A_row)
            b_le.append(b_val)
            A_le.append([-c for c in A_row])
            b_le.append(-b_val)
        else: # '<='
            A_le.append(A_row)
            b_le.append(b_val)
    
    num_constraints_le = len(b_le)
    slack_vars = [Symbol(f"w{i+1}") for i in range(num_constraints_le)]
    
    # ===================================================================================
    # BƯỚC 3: PHA 1 - TÌM NGHIỆM KHẢ THI BAN ĐẦU
    # ===================================================================================
    x0_sym = Symbol('x0')
    tableau_p1: Dict[str, Expr] = {}
    basic_vars_p1: List[str] = [str(s) for s in slack_vars]
    non_basic_vars_p1 = [str(s) for s in tableau_decision_symbols]
    
    const_terms_p1 = {}
    is_initial_feasible = True
    for i in range(num_constraints_le):
        w_var = slack_vars[i]
        lhs_expr = sum(S(A_le[i][j]) * tableau_decision_symbols[j] for j in range(len(tableau_decision_symbols)))
        tableau_p1[str(w_var)] = S(b_le[i]) - lhs_expr
        const_terms_p1[str(w_var)] = b_le[i]
        if b_le[i] < -SIMPLEX_TOLERANCE:
            is_initial_feasible = False

    steps_p1 = {}
    if is_initial_feasible:
        # Nghiệm ban đầu đã khả thi, Pha 1 hoàn thành ngay lập tức.
        title_p1_initial = 'Phase 1 - Bước 0 (Bảng khả thi ban đầu)'
        # Hiển thị bảng ban đầu như là kết quả của một Pha 1 tầm thường.
        tableau_p1_display = tableau_p1.copy()
        # Thêm mục tiêu giả W=0 để hiển thị nhất quán.
        tableau_p1_display['W'] = S.Zero 
        steps_p1[title_p1_initial] = {
            'tableau': _get_ordered_tableau_for_history(tableau_p1_display, 'W')
        }
        
        final_tableau_p1 = tableau_p1
        final_basic_p1 = basic_vars_p1
        final_non_basic_p1 = non_basic_vars_p1
    else:
        # Cần chạy Pha 1 để tìm nghiệm khả thi
        p1_obj_name = 'W'
        for w_var_str in basic_vars_p1:
            tableau_p1[w_var_str] += x0_sym
        tableau_p1[p1_obj_name] = x0_sym
        non_basic_vars_p1.append(str(x0_sym))

        title_step0 = 'Phase 1 - Bước 0 (Bảng khởi tạo)'
        steps_p1[title_step0] = {'tableau': _get_ordered_tableau_for_history(tableau_p1, p1_obj_name)}
        
        leaving_var_p0 = min(const_terms_p1, key=const_terms_p1.get)
        
        pivot_row_expr = tableau_p1[leaving_var_p0]
        expr_for_x0 = simplify(Symbol(leaving_var_p0) - (pivot_row_expr - x0_sym))
        
        new_tableau_p1 = {var: simplify(expr.subs(x0_sym, expr_for_x0)) for var, expr in tableau_p1.items() if var != leaving_var_p0}
        new_tableau_p1[str(x0_sym)] = expr_for_x0
        tableau_p1 = new_tableau_p1

        basic_vars_p1.remove(leaving_var_p0)
        basic_vars_p1.append(str(x0_sym))
        non_basic_vars_p1.remove(str(x0_sym))
        non_basic_vars_p1.append(leaving_var_p0)
        
        title_step1 = f'Phase 1 - Bước 1 (Sau xoay khởi đầu, Vào: {str(x0_sym)}, Ra: {leaving_var_p0})'
        steps_p1[title_step1] = {'tableau': _get_ordered_tableau_for_history(tableau_p1, p1_obj_name)}
        
        status_p1, w_min_val, final_tableau_p1, steps_p1_solver, final_basic_p1, final_non_basic_p1, _ = _simplex_core_solver(
            tableau_p1, basic_vars_p1, non_basic_vars_p1, p1_obj_name, "Phase 1", start_step=2
        )
        steps_p1.update(steps_p1_solver)
        
        if status_p1 not in ['Optimal', 'MultipleOptimal'] or (w_min_val is not None and w_min_val > SIMPLEX_TOLERANCE):
            error_msg = f"Pha 1 kết thúc với min(x0) = {w_min_val:.4f} > 0. Bài toán gốc VÔ NGHIỆM."
            return {'status': status_map_vn['Infeasible'], 'z': 'N/A', 'solution': {}, 'steps': steps_p1, 'error_message': error_msg}

    # ===================================================================================
    # BƯỚC 4: PHA 2 - TÌM NGHIỆM TỐI ƯU
    # ===================================================================================
    tableau_p2: Dict[str, Expr] = {}

    if str(x0_sym) in final_basic_p1:
        x0_row_expr = final_tableau_p1[str(x0_sym)]
        pivot_col_sym = None
        # Ưu tiên xoay một biến quyết định vào cơ sở
        sorted_nb_vars_p2 = sorted(final_non_basic_p1, key=lambda v: get_bland_key(Symbol(v))[0])

        for nb_var_str in sorted_nb_vars_p2:
            if nb_var_str == str(x0_sym): continue
            if abs(x0_row_expr.coeff(Symbol(nb_var_str))) > SIMPLEX_TOLERANCE:
                pivot_col_sym = Symbol(nb_var_str); break
        
        if pivot_col_sym:
            coeff = x0_row_expr.coeff(pivot_col_sym)
            expr_for_pivot_col = simplify((-(x0_row_expr - coeff * pivot_col_sym - x0_row_expr.as_coeff_Add()[0])) / coeff)
            
            for var, expr in final_tableau_p1.items():
                final_tableau_p1[var] = simplify(expr.subs(pivot_col_sym, expr_for_pivot_col))
            
            final_tableau_p1[str(pivot_col_sym)] = expr_for_pivot_col
            final_basic_p1.append(str(pivot_col_sym))
            final_non_basic_p1.remove(str(pivot_col_sym))
    
    for var, expr in final_tableau_p1.items():
        if var not in ['W', str(x0_sym)]:
            tableau_p2[var] = expr.subs(x0_sym, 0)

    basic_vars_p2 = [v for v in final_basic_p1 if v != str(x0_sym)]
    non_basic_vars_p2 = [v for v in final_non_basic_p1 if v != str(x0_sym)]
    
    p2_obj_name = 'z'
    z_expr_orig = sum(S(c_std[i]) * tableau_decision_symbols[i] for i in range(len(c_std)))
    z_expr_to_solve = -z_expr_orig if is_max_problem else z_expr_orig
    
    subs_dict_p2 = {Symbol(bv): tableau_p2[bv] for bv in basic_vars_p2 if Symbol(bv) in z_expr_to_solve.free_symbols and bv in tableau_p2}
    tableau_p2[p2_obj_name] = simplify(z_expr_to_solve.subs(subs_dict_p2))

    combined_steps = steps_p1.copy()
    title_p2_step0 = 'Phase 2 - Bước 0 (Bảng khởi tạo từ Pha 1)'
    combined_steps[title_p2_step0] = {'tableau': _get_ordered_tableau_for_history(tableau_p2, p2_obj_name)}

    status_p2, z_min_value, final_tableau_p2, steps_p2_solver, final_basic_p2, final_non_basic_p2, info_p2 = _simplex_core_solver(
        tableau_p2, basic_vars_p2, non_basic_vars_p2, p2_obj_name, "Phase 2", start_step=1
    )
    combined_steps.update(steps_p2_solver)
    
    # ===================================================================================
    # BƯỚC 5: XỬ LÝ VÀ TRẢ VỀ KẾT QUẢ
    # ===================================================================================
    final_z_val = "N/A"
    if status_p2 in ['Optimal', 'MultipleOptimal'] and z_min_value is not None:
        final_z_val = -z_min_value if is_max_problem else z_min_value
    elif status_p2 == 'Unbounded':
        final_z_val = float('inf') if is_max_problem else float('-inf')

    solution_final = {}
    if status_p2 in ['Optimal', 'MultipleOptimal']:
        parametric_vars = info_p2.get('parametric_vars', [])
        solution_final = _reconstruct_final_solution(
            final_tableau_p2, final_basic_p2, final_non_basic_p2, 
            original_var_info_map, status_p2, parametric_vars
        )

    return {
        'status': status_map_vn.get(status_p2, status_p2), 
        'z': f"{final_z_val:.2f}" if isinstance(final_z_val, (float, int)) else str(final_z_val),
        'solution': solution_final, 
        'steps': combined_steps, 
        'error_message': None
    }
