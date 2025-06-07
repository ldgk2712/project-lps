import logging
from typing import List, Dict, Tuple, Any, Union
from sympy import simplify, solve, Symbol, S, sympify, Add, Mul, Number

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dung sai cho các phép so sánh số thực
SIMPLEX_TOLERANCE = 1e-9

def format_expression_for_printing(expression):
    """Định dạng biểu thức SymPy để in, làm tròn và sắp xếp các số hạng."""
    if not isinstance(expression, (Add, Mul, Symbol, Number)):
        try:
            expression = sympify(expression)
        except (SyntaxError, TypeError):
            return str(expression)

    expression = simplify(expression)
    
    if not expression.free_symbols:
        try:
            if isinstance(expression, Number):
                 num_val = float(expression.evalf(chop=True)) 
                 return f"{round(num_val, 10):.2f}" 
            else: 
                 return str(expression)
        except (TypeError, ValueError): 
             return str(expression)

    const_term = expression.as_coeff_Add()[0] 
    var_terms_expr = expression - const_term 
    
    try:
        const_val = float(const_term.evalf(chop=True))
        const_str = f"{round(const_val, 10):.2f}" if const_val != 0 else "0.00"
    except TypeError:
        const_str = str(const_term) 

    if var_terms_expr == 0: 
        return const_str

    var_term_dict = {}
    terms_to_process = var_terms_expr.as_ordered_terms() if isinstance(var_terms_expr, Add) else [var_terms_expr]

    for term in terms_to_process:
        coeff, var_part = term.as_coeff_Mul() 
        var_str = str(var_part) if var_part != 1 else "" 
        if var_str: 
            var_term_dict[var_str] = var_term_dict.get(var_str, S.Zero) + coeff
            
    sorted_vars = sorted(var_term_dict.keys(), key=lambda x_key: (
        0 if x_key.startswith('x') and not (x_key.endswith('_p') or x_key.endswith('_n')) else
        1 if x_key.startswith('y') and not (x_key.endswith('_p') or x_key.endswith('_n')) else
        2 if x_key.startswith('w') else
        3 if x_key.endswith('_p') or x_key.endswith('_n') else 
        4, x_key
    ))

    var_str_list = []
    for var_key in sorted_vars:
        coeff = var_term_dict[var_key]
        try:
            coeff_val = float(coeff.evalf(chop=True))
            coeff_rounded = round(coeff_val, 2) 
        except TypeError: 
            if coeff == 1: coeff_rounded = 1.0
            elif coeff == -1: coeff_rounded = -1.0
            else: 
                sign_char = ""
                if hasattr(coeff, 'is_negative') and coeff.is_negative: sign_char = "-"
                elif hasattr(coeff, 'is_positive') and coeff.is_positive: sign_char = "+"
                
                term_str_sym = f"{var_key}" if abs(coeff) == 1 else f"{abs(coeff)}*{var_key}"
                var_str_list.append((sign_char, term_str_sym, not (hasattr(coeff, 'is_negative') and coeff.is_negative)))
                continue

        if abs(coeff_rounded) < SIMPLEX_TOLERANCE / 100: 
            continue
        
        sign = "+" if coeff_rounded > 0 else "-"
        abs_coeff_display = abs(coeff_rounded)

        if abs(abs_coeff_display - 1.0) < SIMPLEX_TOLERANCE and var_key: 
            term_str = f"{var_key}"
        else:
            term_str = f"{abs_coeff_display:.2f}*{var_key}"
        
        var_str_list.append((sign, term_str, coeff_rounded > 0))

    if not var_str_list: 
        return const_str

    result_str = ""
    if const_str != "0.00" or not var_str_list:
        result_str = const_str
    
    for i, (sign, term_s, is_positive) in enumerate(var_str_list):
        if result_str and result_str != "0.00" : 
            result_str += f" {sign} {term_s}"
        elif result_str == "0.00": 
            result_str = f"{sign} {term_s}".strip() 
            if result_str.startswith("+"): result_str = result_str[1:].strip()
        elif not result_str: 
            result_str = f"{sign} {term_s}".strip()
            if result_str.startswith("+"): result_str = result_str[1:].strip()

    return result_str if result_str else "0.00"

def _print_tableau(title: str, exprs: Dict[str, Any], basic_vars: List[str] = None, non_basic_vars: List[str] = None) -> None:
    """In bảng Simplex với định dạng cải tiến."""
    print(f"\n{'=' * 70}")
    print(f"{title:^70}")
    print(f"{'-' * 70}")
    
    ordered_var_names = []
    if 'z' in exprs:
        ordered_var_names.append('z')
    
    all_vars_in_exprs = [k for k in exprs if k != 'z']
    all_vars_in_exprs.sort(key=get_bland_key)
    ordered_var_names.extend(all_vars_in_exprs)

    print(f"{'Biến':<15} | {'Biểu thức':<52}") 
    print(f"{'-' * 15} | {'-' * 52}")

    for var_name in ordered_var_names:
        if var_name in exprs:
            expression_to_format = exprs[var_name]
            if not isinstance(expression_to_format, (Add, Mul, Symbol, Number, type(S.Zero), type(S.One))):
                 try:
                    expression_to_format = sympify(expression_to_format)
                 except (SyntaxError, TypeError, AttributeError):
                    pass
            
            formatted_str = format_expression_for_printing(expression_to_format)
            print(f"{var_name:<15} | {formatted_str:<52}")
            
    if basic_vars or non_basic_vars:
        print(f"{'-' * 70}")
        if basic_vars:
            print(f"Biến cơ sở    : {', '.join(sorted(basic_vars, key=get_bland_key))}")
        if non_basic_vars:
            print(f"Biến không cơ bản: {', '.join(sorted(non_basic_vars, key=get_bland_key))}")
    print(f"{'=' * 70}")

def get_bland_key(var_obj: Union[str, Symbol]):
    """Tạo khóa để sắp xếp các biến theo quy tắc Bland."""
    var_name_str = str(var_obj)
    type_priority = 99
    main_index = 9999
    sub_index = 0

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
    elif base_name.startswith('a'):
        type_priority = 2

    try:
        numeric_part = ''.join(filter(str.isdigit, base_name))
        main_index = int(numeric_part) if numeric_part else (0 if len(base_name) == 1 else 9998)
    except ValueError:
        pass

    return (type_priority, main_index, sub_index, var_name_str)

def _get_current_solution_values(
    current_tableau: Dict[str, Any], 
    basic_var_names: List[str],
    non_basic_var_names: List[str],
    original_var_info_map: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Tính toán giá trị số của các biến quyết định gốc tại một bước nhất định."""
    
    solution_values_numeric = {}
    subs_dict = {Symbol(nb): 0 for nb in non_basic_var_names}

    tableau_vars_values = {}
    for b_var in basic_var_names:
        val = 0.0
        if b_var in current_tableau:
            expr_val = current_tableau[b_var].subs(subs_dict)
            if expr_val.is_Number:
                try:
                    val = float(expr_val.evalf())
                except (TypeError, ValueError):
                    val = 0.0
        tableau_vars_values[b_var] = val
    
    for nb_var in non_basic_var_names:
        tableau_vars_values[nb_var] = 0.0

    for info in original_var_info_map:
        original_var_name = info['original_name']
        
        if info.get('is_standard', False):
            tableau_name = info['tableau_name']
            solution_values_numeric[original_var_name] = tableau_vars_values.get(tableau_name, 0.0)
        
        elif info.get('is_transformed_neg', False):
            tableau_name = info['tableau_name']
            solution_values_numeric[original_var_name] = -tableau_vars_values.get(tableau_name, 0.0)
            
        elif info.get('is_urs', False):
            pos_name = info['pos_name']
            neg_name = info['neg_name']
            pos_val = tableau_vars_values.get(pos_name, 0.0)
            neg_val = tableau_vars_values.get(neg_name, 0.0)
            solution_values_numeric[original_var_name] = pos_val - neg_val
            
    return solution_values_numeric

def _simplex_min(A: List[List[float]], b: List[float], c: List[float], 
                 constraint_types: List[str], variable_types: List[str], 
                 objective_type: str, original_var_info_map_outer: List[Dict[str, Any]]) -> Tuple[str, Union[float, None], Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]:
    """Thuật toán Simplex cốt lõi, trả về lịch sử các bước với tọa độ số."""
    m, n_original = len(A), len(A[0]) 

    A_std = [row[:] for row in A] 
    b_std = b[:]                   

    tableau_decision_symbols = [] 
    c_tableau = []                
    original_var_info_map = [] 
    
    current_var_idx_map = {}

    for i in range(n_original):
        original_name = f"x{i+1}" 
        var_type = variable_types[i].strip()
        
        if var_type == '<=0': 
            y_idx = current_var_idx_map.get('y', 0) + 1
            current_var_idx_map['y'] = y_idx
            tableau_name = f"y{y_idx}" 
            symbol = Symbol(tableau_name)
            tableau_decision_symbols.append(symbol)
            c_tableau.append(-c[i]) 
            original_var_info_map.append({
                'original_name': original_name, 'tableau_name': tableau_name, 
                'symbol': symbol, 'is_transformed_neg': True, 'original_idx': i
            })
        elif var_type == 'URS': 
            pos_name = f"x{i+1}_p" 
            neg_name = f"x{i+1}_n" 
            pos_symbol, neg_symbol = Symbol(pos_name), Symbol(neg_name)
            tableau_decision_symbols.extend([pos_symbol, neg_symbol])
            c_tableau.extend([c[i], -c[i]]) 
            original_var_info_map.append({
                'original_name': original_name, 'pos_name': pos_name, 'neg_name': neg_name,
                'pos_symbol': pos_symbol, 'neg_symbol': neg_symbol, 
                'is_urs': True, 'original_idx': i
            })
        else: # Default '>=0'
            tableau_name = original_name 
            symbol = Symbol(tableau_name)
            tableau_decision_symbols.append(symbol)
            c_tableau.append(c[i])
            original_var_info_map.append({
                'original_name': original_name, 'tableau_name': tableau_name, 
                'symbol': symbol, 'is_standard': True, 'original_idx': i
            })

    A_tableau_cols = []
    for i in range(n_original):
        if variable_types[i].strip() == '<=0':
            A_tableau_cols.append([-A_std[j][i] for j in range(m)])
        elif variable_types[i].strip() == 'URS':
            A_tableau_cols.append([A_std[j][i] for j in range(m)])
            A_tableau_cols.append([-A_std[j][i] for j in range(m)])
        else:
            A_tableau_cols.append([A_std[j][i] for j in range(m)])

    A_tableau = [[A_tableau_cols[c][r] for c in range(len(A_tableau_cols))] for r in range(m)]

    slack_symbols = [Symbol(f"w{j+1}") for j in range(m)] 
    
    step = 0 
    steps_history: Dict[str, Dict[str, Any]] = {}
    current_tableau: Dict[str, Any] = {}

    basic_var_names = [str(s) for s in slack_symbols]
    non_basic_var_names = [str(s) for s in tableau_decision_symbols]

    z_expr = S.Zero
    for j, sym in enumerate(tableau_decision_symbols):
        z_expr += S(c_tableau[j]) * sym
    current_tableau['z'] = simplify(z_expr)

    for i in range(m): 
        row_expr = S(b_std[i])
        for j, sym in enumerate(tableau_decision_symbols):
            row_expr -= S(A_tableau[i][j]) * sym
        current_tableau[slack_symbols[i].name] = simplify(row_expr)
    
    title_step_0 = 'Bước 0 (Bảng khởi tạo - Initial Tableau)'
    current_solution_step_0 = _get_current_solution_values(current_tableau, basic_var_names, non_basic_var_names, original_var_info_map_outer)
    coords_step_0 = [current_solution_step_0.get('x1', 0.0), current_solution_step_0.get('x2', 0.0)]
    steps_history[title_step_0] = {
        'tableau': {k: v.copy() if hasattr(v, 'copy') else v for k, v in current_tableau.items()},
        'coords': coords_step_0
    }
    _print_tableau(title_step_0, current_tableau, basic_var_names, non_basic_var_names)
    
    status = 'Processing' 
    max_iterations = 100 
    iteration_count = 0
    parameter_names_for_multiple_optima: List[str] = []

    while iteration_count < max_iterations:
        iteration_count += 1
        z_row_expr = current_tableau['z']
        
        is_degenerate = any(
            abs(float(current_tableau[b_var].subs({Symbol(s_nb): 0 for s_nb in non_basic_var_names}).evalf(chop=True))) < SIMPLEX_TOLERANCE
            for b_var in basic_var_names
        )
        if is_degenerate:
            logger.info("Phát hiện suy biến.")

        # ----- SỬA LỖI: HỢP NHẤT LOGIC CHỌN BIẾN VÀO -----
        # Luôn sử dụng Quy tắc Bland để chọn biến vào một cách nhất quán để chống lặp.
        entering_var_name: Union[str, None] = None
        sorted_candidates = sorted(non_basic_var_names, key=get_bland_key)
        for var_cand in sorted_candidates:
            coeff = z_row_expr.coeff(Symbol(var_cand))
            if isinstance(coeff, Number) and float(coeff.evalf()) < -SIMPLEX_TOLERANCE:
                entering_var_name = var_cand
                logger.info(f"Quy tắc Bland: Chọn biến vào {entering_var_name}.")
                break  # Dừng ngay khi tìm thấy biến đầu tiên thỏa mãn

        if entering_var_name is None:
            status = 'Optimal'
            logger.info(f"Trạng thái cuối cùng: {status}")
            break

        entering_var_symbol = Symbol(entering_var_name) 
        
        potential_leaving_vars_with_ratio: List[Tuple[float, str]] = []
        found_positive_pivot = False
        for b_var in basic_var_names:
            constraint_expr = current_tableau[b_var]
            pivot_coeff = -constraint_expr.coeff(entering_var_symbol)
            if isinstance(pivot_coeff, Number) and float(pivot_coeff.evalf()) > SIMPLEX_TOLERANCE:
                found_positive_pivot = True
                rhs_val = float(constraint_expr.subs({Symbol(nb): 0 for nb in non_basic_var_names}).evalf())
                if rhs_val >= -SIMPLEX_TOLERANCE:
                    ratio = rhs_val / float(pivot_coeff.evalf())
                    potential_leaving_vars_with_ratio.append((ratio, b_var))

        if not found_positive_pivot:
            status = 'Unbounded'
            break

        # Chọn biến ra theo quy tắc Bland (chỉ số nhỏ nhất nếu có nhiều min_ratio bằng nhau)
        min_ratio = min(r for r, v in potential_leaving_vars_with_ratio)
        tied_candidates = [v for r, v in potential_leaving_vars_with_ratio if abs(r - min_ratio) < SIMPLEX_TOLERANCE]
        tied_candidates.sort(key=get_bland_key)
        leaving_var_name = tied_candidates[0]

        step += 1
        pivot_row_expr = current_tableau[leaving_var_name]
        coeff_entering_in_pivot = pivot_row_expr.coeff(entering_var_symbol)
        
        new_entering_expr = (Symbol(leaving_var_name) - (pivot_row_expr - coeff_entering_in_pivot * entering_var_symbol)) / coeff_entering_in_pivot
        
        new_tableau_temp = {entering_var_name: simplify(new_entering_expr)}
        for var, old_expr in current_tableau.items():
            if var != leaving_var_name:
                new_tableau_temp[var] = simplify(old_expr.subs(entering_var_symbol, new_entering_expr))

        current_tableau = new_tableau_temp
        
        basic_var_names.remove(leaving_var_name)
        basic_var_names.append(entering_var_name)
        non_basic_var_names.remove(entering_var_name)
        non_basic_var_names.append(leaving_var_name)

        title_step_n = f"Bước {step} (Vào: {entering_var_name}, Ra: {leaving_var_name})"
        current_solution_step_n = _get_current_solution_values(current_tableau, basic_var_names, non_basic_var_names, original_var_info_map_outer)
        coords_step_n = [current_solution_step_n.get('x1', 0.0), current_solution_step_n.get('x2', 0.0)]
        steps_history[title_step_n] = {
            'tableau': {k: v.copy() if hasattr(v, 'copy') else v for k, v in current_tableau.items()},
            'coords': coords_step_n
        }
        _print_tableau(title_step_n, current_tableau, basic_var_names, non_basic_var_names)

    z_star_value: Union[float, None] = None
    solution_values: Dict[str, Any] = {}
    if status in ['Optimal', 'Multiple']:
        z_final_expr = current_tableau.get('z', S.Zero)
        subs_dict = {Symbol(nb): 0 for nb in non_basic_var_names if nb not in parameter_names_for_multiple_optima}
        z_final_expr_subbed = z_final_expr.subs(subs_dict)
        if not z_final_expr_subbed.free_symbols:
            z_star_value = float(z_final_expr_subbed.evalf(chop=True))

        for info in original_var_info_map_outer:
            solution_values[info['original_name']] = _get_current_solution_values(current_tableau, basic_var_names, non_basic_var_names, [info]).get(info['original_name'], S.Zero)

    elif status == 'Unbounded':
        z_star_value = float('-inf')

    return status, z_star_value, solution_values, steps_history, parameter_names_for_multiple_optima

def auto_simplex(
    A: List[List[float]], 
    b: List[float],       
    c: List[float],       
    constraint_types: List[str], 
    objective_type: str = 'max',
    variable_types: List[str] | None = None, 
) -> Dict[str, Any]:
    """Hàm chính, kiểm tra dạng chuẩn trước khi giải."""
    
    # --- BẮT ĐẦU: KIỂM TRA DẠNG CHUẨN TRƯỚC KHI GIẢI ---
    is_not_standard = False
    for i, ct in enumerate(constraint_types):
        if ct.strip() != '<=':
            is_not_standard = True
            logger.warning(f"Phát hiện ràng buộc #{i+1} dạng '{ct}'. Đơn hình chuẩn yêu cầu dạng '<='.")
            break
            
    if not is_not_standard:
        for i, val in enumerate(b):
            if val < 0:
                is_not_standard = True
                logger.warning(f"Phát hiện vế phải âm ({val}) ở ràng buộc #{i+1}. Đơn hình chuẩn yêu cầu vế phải không âm.")
                break

    if is_not_standard:
        error_msg = ("Bài toán này không thể giải bằng đơn hình chuẩn vì không ở dạng chính tắc (yêu cầu tất cả ràng buộc là '<=' và vế phải không âm). Vui lòng chọn thuật toán Đơn hình Hai Pha để giải.")
        return {
            'status': 'Cần dùng phương pháp Hai Pha',
            'z': "N/A",
            'solution': {},
            'steps': {'Lỗi': {'tableau': [('Thông báo', error_msg)], 'coords': None}},
            'error_message': error_msg,
            'parameter_conditions': ""
        }
    # --- KẾT THÚC: KIỂM TRA DẠNG CHUẨN ---

    num_vars_orig = len(c)
    if variable_types is None:
        variable_types = ['>=0'] * num_vars_orig 

    # Tạo map thông tin biến gốc để truyền vào _simplex_min
    original_var_info_map_for_coords = []
    for i in range(num_vars_orig):
        original_name = f"x{i+1}" 
        var_type = variable_types[i].strip()
        info = {'original_name': original_name, 'type': var_type}
        if var_type == '<=0':
             info.update({'tableau_name': f"y{i+1}", 'is_transformed_neg': True})
        elif var_type == 'URS':
             info.update({'pos_name': f"x{i+1}_p", 'neg_name': f"x{i+1}_n", 'is_urs': True})
        else: # '>=0'
            info.update({'tableau_name': original_name, 'is_standard': True})
        original_var_info_map_for_coords.append(info)

    c_effective = c[:] 
    is_maximization = objective_type.strip().lower().startswith('max')
    if is_maximization:
        c_effective = [-ci for ci in c_effective]

    try:
        status_solver, z_star_from_solver, sol_vals_sympy_from_solver, steps_history_raw, _ = _simplex_min(
            A, b, c_effective, 
            constraint_types,
            variable_types, 
            "min",
            original_var_info_map_for_coords
        )
    except Exception as e:
        logger.error(f"Lỗi không mong muốn trong quá trình giải Simplex: {str(e)}", exc_info=True)
        return {'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {}, 'error_message': f"Lỗi hệ thống: {str(e)}"}
    
    formatted_steps_output: Dict[str, Dict[str, Any]] = {}
    for title, step_data_dict in steps_history_raw.items():
        tableau_sympy = step_data_dict.get('tableau', {})
        coords = step_data_dict.get('coords')
        
        ordered_tableau_list = []
        # Logic sắp xếp và định dạng bảng...
        all_vars_in_step = sorted(list(tableau_sympy.keys()), key=get_bland_key)
        if 'z' in all_vars_in_step:
            all_vars_in_step.insert(0, all_vars_in_step.pop(all_vars_in_step.index('z')))

        for var_name in all_vars_in_step:
            formatted_expr = format_expression_for_printing(tableau_sympy[var_name])
            ordered_tableau_list.append((var_name, formatted_expr))

        formatted_steps_output[title] = {
            'tableau': ordered_tableau_list,
            'coords': coords
        }

    z_final_display: Union[float, str] = "N/A"
    if status_solver in ['Optimal', 'Multiple']:
        if z_star_from_solver is not None:
            z_final_display = -z_star_from_solver if is_maximization else z_star_from_solver
    elif status_solver == 'Unbounded':
        z_final_display = float('inf') if is_maximization else float('-inf')

    solution_output_formatted: Dict[str, str] = {
        var: format_expression_for_printing(val) for var, val in sol_vals_sympy_from_solver.items()
    }

    status_map_vn = {'Optimal': 'Tối ưu (Optimal)', 'Multiple': 'Vô số nghiệm', 'Unbounded': 'Không giới nội', 'Error': 'Lỗi'}
    final_status = status_map_vn.get(status_solver, status_solver)

    return {
        'status': final_status,
        'z': f"{z_final_display:.2f}" if isinstance(z_final_display, float) else str(z_final_display),
        'solution': solution_output_formatted, 
        'steps': formatted_steps_output,
        'error_message': None,
        'parameter_conditions': ""
    }
