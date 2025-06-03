import logging
from typing import List, Dict, Tuple, Any, Union
from sympy import simplify, solve, Symbol, S, sympify, Add, Mul, Number

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a tolerance for floating point comparisons
SIMPLEX_TOLERANCE = 1e-9

def format_expression_for_printing(expression):
    """Định dạng biểu thức SymPy với hằng số đứng trước và các biến được sắp xếp, làm tròn đến 2 chữ số thập phân."""
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
    all_vars_in_exprs.sort(key=get_bland_key)  # Sử dụng get_bland_key để hiển thị nhất quán

    ordered_var_names.extend(all_vars_in_exprs)
    
    for k_expr in exprs.keys():
        if k_expr not in ordered_var_names:
            ordered_var_names.append(k_expr)

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
            sorted_basic_vars = basic_vars[:]
            sorted_basic_vars.sort(key=get_bland_key)
            print(f"Biến cơ bản    : {', '.join(sorted_basic_vars)}")
        if non_basic_vars:
            sorted_non_basic_vars = non_basic_vars[:]
            sorted_non_basic_vars.sort(key=get_bland_key)
            print(f"Biến không cơ bản: {', '.join(sorted_non_basic_vars)}")
    print(f"{'=' * 70}")

def get_bland_key(var_obj: Union[str, Symbol]):
    """
    Tạo khóa để sắp xếp các biến theo quy tắc Bland.
    Ưu tiên: Biến quyết định (x, y) > Biến bù (w) > Biến nhân tạo (a).
    Trong mỗi nhóm, chỉ số nhỏ hơn được ưu tiên.
    """
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
        try:
            main_index = int(base_name[1:])
        except ValueError:
            main_index = 0
    elif base_name.startswith('w'):
        type_priority = 1
        try:
            main_index = int(base_name[1:])
        except ValueError:
            main_index = 0
    elif base_name.startswith('a'):
        type_priority = 2
        try:
            main_index = int(base_name[1:])
        except ValueError:
            main_index = 0
    else:
        try:
            main_index = int(''.join(filter(str.isdigit, base_name)))
        except ValueError:
            pass

    return (type_priority, main_index, sub_index, var_name_str)

def _simplex_min(A: List[List[float]], b: List[float], c: List[float], 
                 constraint_types: List[str], variable_types: List[str], 
                 objective_type: str) -> Tuple[str, Union[float, None], Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Thuật toán Simplex cốt lõi cho bài toán min, sử dụng Quy tắc Bland khi phát hiện suy biến.
    Trả về: status, z_star, solution_values, steps_history, parameter_names_for_multiple_optima
    """
    m, n_original = len(A), len(A[0]) 

    if len(b) != m or len(constraint_types) != m or len(c) != n_original or len(variable_types) != n_original:
        raise ValueError("Kích thước đầu vào không khớp (A, b, c, constraint_types, variable_types).")
    for i, ct in enumerate(constraint_types):
        if ct.strip() not in ['<=', '>=', '=']: 
            raise ValueError(f"Loại ràng buộc không hợp lệ tại vị trí {i}: {ct}. Chỉ chấp nhận '<=', '>=', '='.")

    A_std = [row[:] for row in A] 
    b_std = b[:]                   

    tableau_decision_symbols = [] 
    c_tableau = []                
    A_tableau = [[] for _ in range(m)] 
    original_var_info_map = [] 

    current_var_idx = 0 
    
    for i in range(n_original):
        original_name = f"x{i+1}" 
        var_type = variable_types[i].strip()
        
        if var_type == '<=0': 
            tableau_name = f"y{current_var_idx+1}" 
            current_var_idx +=1
            symbol = Symbol(tableau_name)
            tableau_decision_symbols.append(symbol)
            c_tableau.append(-c[i]) 
            for j_row in range(m): 
                A_tableau[j_row].append(-A_std[j_row][i]) 
            original_var_info_map.append({
                'original_name': original_name, 'tableau_name': tableau_name, 
                'symbol': symbol, 'is_transformed_neg': True, 'original_idx': i
            })
        elif var_type == 'URS': 
            pos_name = f"x{i+1}_p" 
            neg_name = f"x{i+1}_n" 
            pos_symbol = Symbol(pos_name)
            neg_symbol = Symbol(neg_name)
            tableau_decision_symbols.extend([pos_symbol, neg_symbol])
            c_tableau.extend([c[i], -c[i]]) 
            for j_row in range(m): 
                A_tableau[j_row].extend([A_std[j_row][i], -A_std[j_row][i]])
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
            for j_row in range(m):
                A_tableau[j_row].append(A_std[j_row][i])
            original_var_info_map.append({
                'original_name': original_name, 'tableau_name': tableau_name, 
                'symbol': symbol, 'is_standard': True, 'original_idx': i
            })

    slack_symbols = [Symbol(f"w{j+1}") for j in range(m)] 
    for constraint_idx in range(m): 
        for r_idx in range(m): 
            A_tableau[r_idx].append(1 if r_idx == constraint_idx else 0)
        c_tableau.append(0) 

    all_tableau_symbols = tableau_decision_symbols + slack_symbols 
    num_vars_in_tableau = len(all_tableau_symbols)
    
    step = 0 
    steps_history: Dict[str, Dict[str, Any]] = {} 
    current_tableau: Dict[str, Any] = {} 

    basic_var_names = [str(s) for s in slack_symbols]
    non_basic_var_names = [str(s) for s in tableau_decision_symbols]

    z_expr = S.Zero
    for j_col in range(len(c_tableau)): 
        z_expr += S(c_tableau[j_col]) * all_tableau_symbols[j_col]
    current_tableau['z'] = simplify(z_expr)

    for i_row in range(m): 
        row_expr_for_basic_var = S(b_std[i_row])
        idx_A_col = 0
        for decision_sym in tableau_decision_symbols: 
            if idx_A_col < len(A_tableau[i_row]) - m : 
                 row_expr_for_basic_var -= S(A_tableau[i_row][idx_A_col]) * decision_sym
            idx_A_col +=1
        current_tableau[slack_symbols[i_row].name] = simplify(row_expr_for_basic_var)
    
    title_step_0 = 'Bước 0 (Bảng khởi tạo - Initial Tableau)'
    current_tableau_for_history = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in current_tableau.items()}
    steps_history[title_step_0] = current_tableau_for_history
    _print_tableau(title_step_0, current_tableau, basic_var_names, non_basic_var_names)
    
    status = 'Processing' 
    max_iterations = 100
    iteration_count = 0
    parameter_names_for_multiple_optima: List[str] = []


    while iteration_count < max_iterations:
        iteration_count += 1
        z_row_expr = current_tableau['z']
        
        is_degenerate = False
        for b_var_str in basic_var_names:
            rhs_val = current_tableau[b_var_str].subs({Symbol(s): 0 for s in non_basic_var_names})
            if isinstance(rhs_val, Number) and abs(float(rhs_val.evalf())) < SIMPLEX_TOLERANCE:
                is_degenerate = True
                logger.info(f"Phát hiện suy biến tại {b_var_str} với RHS ≈ 0")
                break

        entering_var_name: Union[str, None] = None
        if is_degenerate:
            sorted_candidates_bland = sorted(all_tableau_symbols, key=get_bland_key)
            for var_sym_candidate in sorted_candidates_bland: # Bland uses all_tableau_symbols sorted
                var_str_candidate = str(var_sym_candidate)
                if var_str_candidate in non_basic_var_names:
                    coeff_in_z = z_row_expr.coeff(var_sym_candidate)
                    if isinstance(coeff_in_z, Number) and float(coeff_in_z.evalf()) < -SIMPLEX_TOLERANCE:
                        entering_var_name = var_str_candidate
                        logger.info(f"Suy biến: Chọn biến vào {entering_var_name} theo Quy tắc Bland")
                        break
        else:
            most_negative_coeff_in_z = S.Zero
            # Sort non_basic_var_names by Bland key to break ties for entering variable selection consistently
            sorted_non_basics_for_entering = sorted(non_basic_var_names, key=get_bland_key)
            for nb_var_str in sorted_non_basics_for_entering:
                nb_sym = Symbol(nb_var_str)
                coeff = z_row_expr.coeff(nb_sym)
                if isinstance(coeff, Number) and float(coeff.evalf()) < float(most_negative_coeff_in_z.evalf() - SIMPLEX_TOLERANCE):
                    most_negative_coeff_in_z = coeff
                    entering_var_name = nb_var_str
                # If coefficients are equal, Bland's rule (already applied by sorting) keeps the first one.
            if entering_var_name:
                 logger.info(f"Không suy biến: Chọn biến vào {entering_var_name} với hệ số {most_negative_coeff_in_z}")


        if entering_var_name is None:
            _internal_has_alternative_optima_condition = False
            non_basics_with_zero_coeff_in_z_temp = [] 
            sorted_non_basics_for_check = sorted(non_basic_var_names, key=get_bland_key)
            for nb_var_str_check in sorted_non_basics_for_check:
                coeff_check = z_row_expr.coeff(Symbol(nb_var_str_check))
                if isinstance(coeff_check, Number) and abs(float(coeff_check.evalf())) < SIMPLEX_TOLERANCE:
                    _internal_has_alternative_optima_condition = True
                    non_basics_with_zero_coeff_in_z_temp.append(nb_var_str_check)
            
            if _internal_has_alternative_optima_condition:
                status = 'Multiple' 
                parameter_names_for_multiple_optima = non_basics_with_zero_coeff_in_z_temp[:]
                logger.info(f"Vô số nghiệm: Biến NB có hệ số 0 trong Z: {', '.join(parameter_names_for_multiple_optima)}")
            else:
                status = 'Optimal'
                logger.info("Tìm thấy nghiệm tối ưu")
            break 

        entering_var_symbol = Symbol(entering_var_name) 
        
        leaving_var_name: Union[str, None] = None 
        min_positive_ratio = float('inf') 
        tie_candidates_for_leaving: List[str] = []
        
        found_positive_pivot_coeff_for_entering_var = False
        # Sort basic_var_names by Bland key for ratio test tie-breaking
        sorted_basic_vars_for_ratio_test = sorted(basic_var_names, key=get_bland_key)

        for b_var_str in sorted_basic_vars_for_ratio_test: # Use sorted list
            constraint_expr_for_basic_var = current_tableau[b_var_str] 
            coeff_of_entering_in_basic_var_expr = constraint_expr_for_basic_var.coeff(entering_var_symbol)
            pivot_col_coeff = -coeff_of_entering_in_basic_var_expr

            if float(pivot_col_coeff.evalf()) > SIMPLEX_TOLERANCE:
                found_positive_pivot_coeff_for_entering_var = True
                
                current_rhs_val_expr = constraint_expr_for_basic_var.subs({Symbol(s): 0 for s in non_basic_var_names if s != entering_var_name})
                current_rhs_val = S.Zero
                if hasattr(current_rhs_val_expr, 'is_Number') and current_rhs_val_expr.is_Number:
                    current_rhs_val = float(current_rhs_val_expr.evalf(chop=True))
                elif hasattr(current_rhs_val_expr, 'as_coeff_Add'):
                    const_part, _ = current_rhs_val_expr.as_coeff_Add()
                    current_rhs_val = float(const_part.evalf(chop=True))
                else:
                    logger.warning(f"Không thể xác định RHS số cho {b_var_str}. Bỏ qua.")
                    continue
                
                if current_rhs_val < -SIMPLEX_TOLERANCE: # Should not happen if feasible
                    continue 

                ratio = current_rhs_val / float(pivot_col_coeff.evalf())

                if ratio < min_positive_ratio - SIMPLEX_TOLERANCE:
                    min_positive_ratio = ratio
                    tie_candidates_for_leaving = [b_var_str]
                elif abs(ratio - min_positive_ratio) < SIMPLEX_TOLERANCE:
                    tie_candidates_for_leaving.append(b_var_str)
        
        if not found_positive_pivot_coeff_for_entering_var: 
            status = 'Unbounded'
            title_unbounded = f"Bước {step+1} (Biến vào: {entering_var_name}, Không giới nội)"
            steps_history[title_unbounded] = {k: v.copy() if hasattr(v, 'copy') else v for k,v in current_tableau.items()}
            _print_tableau(title_unbounded, current_tableau, basic_var_names, non_basic_var_names)
            logger.warning(f"Không giới nội tại bước {step+1} (biến vào {entering_var_name})")
            break

        if not tie_candidates_for_leaving: 
            # This case should ideally not be reached if found_positive_pivot_coeff_for_entering_var is true
            # unless all RHS are zero or negative for positive pivot coeffs, which might indicate issues.
            status = 'Infeasible' # Or Error, depending on context
            title_issue = f"Bước {step+1} (Biến vào: {entering_var_name}, Không tìm thấy biến ra hợp lệ)"
            steps_history[title_issue] = {k: v.copy() if hasattr(v, 'copy') else v for k,v in current_tableau.items()}
            _print_tableau(title_issue, current_tableau, basic_var_names, non_basic_var_names)
            logger.warning(f"Không tìm thấy biến ra hợp lệ cho {entering_var_name} tại bước {step+1}")
            break
        
        if tie_candidates_for_leaving:
            # Bland's rule for leaving variable: choose the one with the smallest Bland index.
            tie_candidates_for_leaving.sort(key=get_bland_key) # Explicitly apply Bland to ties
            leaving_var_name = tie_candidates_for_leaving[0]
            logger.info(f"Chọn biến ra {leaving_var_name} (Tỷ lệ: {min_positive_ratio:.4f}, Bland cho hòa nếu có)")

        else: # Should not be reached if logic above is correct
            status = 'Error'
            logger.error(f"Lỗi logic chọn biến ra tại bước {step+1}")
            break


        if leaving_var_name is None: # Should be caught by "not tie_candidates_for_leaving"
            status = 'Error'
            logger.error(f"Lỗi: leaving_var_name là None tại bước {step+1}")
            break

        step += 1
        logger.info(f"--- Pivot Step {step} ---")
        logger.info(f"Biến vào (Entering): {entering_var_name}, Biến ra (Leaving): {leaving_var_name}")

        pivot_row_expr = current_tableau[leaving_var_name]
        coeff_entering_in_pivot_row = pivot_row_expr.coeff(entering_var_symbol)

        if abs(float(coeff_entering_in_pivot_row.evalf(chop=True))) < SIMPLEX_TOLERANCE / 100.0:
            logger.error(f"LỖI: Phần tử pivot {coeff_entering_in_pivot_row} quá gần 0 đối với biến vào '{entering_var_name}' trong dòng của '{leaving_var_name}'.")
            status = 'Error' 
            break 

        P_rest_raw = pivot_row_expr - coeff_entering_in_pivot_row * entering_var_symbol
        numerator_for_new_entering_expr_raw = Symbol(leaving_var_name) - P_rest_raw
        E_expr_for_substitution = numerator_for_new_entering_expr_raw / coeff_entering_in_pivot_row
        new_entering_var_value_expr_simplified = simplify(E_expr_for_substitution)
        
        new_tableau = {}
        new_tableau[entering_var_name] = new_entering_var_value_expr_simplified

        for var_name_in_old_tableau, old_expr_in_tableau in current_tableau.items():
            if var_name_in_old_tableau == leaving_var_name: 
                continue
            if var_name_in_old_tableau == entering_var_name: 
                 continue

            substituted_expr = old_expr_in_tableau.subs(entering_var_symbol, E_expr_for_substitution)
            simplified_new_row_expr = simplify(substituted_expr)
            new_tableau[var_name_in_old_tableau] = simplified_new_row_expr
        
        current_tableau = new_tableau 
        
        basic_var_names.remove(leaving_var_name)
        basic_var_names.append(entering_var_name)
        non_basic_var_names.remove(entering_var_name)
        non_basic_var_names.append(leaving_var_name)
        
        basic_var_names.sort(key=get_bland_key)
        non_basic_var_names.sort(key=get_bland_key)

        title_step_n = f"Bước {step} (Biến vào: {entering_var_name}, Biến ra: {leaving_var_name})"
        current_tableau_for_history = {}
        for k_hist, v_hist in current_tableau.items(): 
            if hasattr(v_hist, 'copy') and callable(v_hist.copy):
                 current_tableau_for_history[k_hist] = v_hist.copy()
            else:
                 current_tableau_for_history[k_hist] = v_hist 

        steps_history[title_step_n] = current_tableau_for_history
        _print_tableau(title_step_n, current_tableau, basic_var_names, non_basic_var_names)

    if iteration_count >= max_iterations:
        logger.warning(f"Đạt tối đa {max_iterations} vòng lặp")
        status = "MaxIterations"

    z_star_value: Union[float, None] = None
    solution_values: Dict[str, Any] = {} 

    if status == 'Optimal' or status == 'Multiple':
        z_final_expr = current_tableau.get('z', S.Zero)
        subs_all_non_basics_to_zero_for_z_val = {Symbol(nb_str): 0 for nb_str in non_basic_var_names if nb_str not in parameter_names_for_multiple_optima}
        
        try:
            z_expr_substituted = z_final_expr.subs(subs_all_non_basics_to_zero_for_z_val)
            if not z_expr_substituted.free_symbols: 
                 z_star_value_eval = z_expr_substituted.evalf(chop=True)
                 z_star_value = float(z_star_value_eval)
        except Exception as e:
            logger.error(f"Không thể tính Z*: {e}")
            z_star_value = None 

        if status != 'Error': 
            subs_for_solution_expr = {} 
            for nb_str in non_basic_var_names:
                if nb_str not in parameter_names_for_multiple_optima:
                     subs_for_solution_expr[Symbol(nb_str)] = 0
            
            for info in original_var_info_map:
                original_var_name = info['original_name']
                val_expr_final_for_original_var = S.Zero 
                is_urs_var = info.get('is_urs', False)
                is_transformed_neg_var = info.get('is_transformed_neg', False)

                if is_urs_var:
                    pos_name = info['pos_name']
                    neg_name = info['neg_name']
                    pos_sym = Symbol(pos_name)
                    neg_sym = Symbol(neg_name)
                    
                    pos_val_or_expr = current_tableau.get(pos_name, pos_sym) if pos_name in basic_var_names else \
                                      (S.Zero if pos_name in non_basic_var_names and pos_name not in parameter_names_for_multiple_optima else pos_sym)
                    neg_val_or_expr = current_tableau.get(neg_name, neg_sym) if neg_name in basic_var_names else \
                                      (S.Zero if neg_name in non_basic_var_names and neg_name not in parameter_names_for_multiple_optima else neg_sym)
                    
                    val_expr_final_for_original_var = (pos_val_or_expr - neg_val_or_expr).subs(subs_for_solution_expr)

                elif is_transformed_neg_var:
                    tableau_name = info['tableau_name']
                    tab_sym = Symbol(tableau_name)
                    tableau_val_or_expr = current_tableau.get(tableau_name, tab_sym) if tableau_name in basic_var_names else \
                                          (S.Zero if tableau_name in non_basic_var_names and tableau_name not in parameter_names_for_multiple_optima else tab_sym)
                    val_expr_final_for_original_var = (-tableau_val_or_expr).subs(subs_for_solution_expr)
                else: 
                    tableau_name = info['tableau_name']
                    tab_sym = Symbol(tableau_name)
                    if tableau_name in basic_var_names:
                        val_expr_final_for_original_var = current_tableau[tableau_name].subs(subs_for_solution_expr)
                    elif tableau_name in non_basic_var_names:
                        if tableau_name in parameter_names_for_multiple_optima: 
                            val_expr_final_for_original_var = tab_sym 
                        else: 
                            val_expr_final_for_original_var = S.Zero
                    else: 
                        val_expr_final_for_original_var = tab_sym 


                solution_values[original_var_name] = simplify(val_expr_final_for_original_var)
            
            if status == 'Multiple':
                is_truly_parametric_for_original_vars = False
                for orig_var_name, sym_expr_val in solution_values.items():
                    if hasattr(sym_expr_val, 'free_symbols'):
                        for free_sym in sym_expr_val.free_symbols:
                            if str(free_sym) in parameter_names_for_multiple_optima:
                                is_truly_parametric_for_original_vars = True
                                break
                    if is_truly_parametric_for_original_vars:
                        break
                if not is_truly_parametric_for_original_vars and parameter_names_for_multiple_optima:
                    logger.info("Vô số nghiệm ở bảng, nhưng nghiệm gốc có thể là duy nhất. Giữ trạng thái 'Multiple'.")

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
    """Hàm chính để giải bài toán quy hoạch tuyến tính bằng Simplex với Quy tắc Bland khi suy biến."""
    num_constraints_orig, num_vars_orig = len(A), len(c)
    tol = SIMPLEX_TOLERANCE 

    if not A or not all(len(row) == num_vars_orig for row in A) or \
       len(b) != num_constraints_orig or len(constraint_types) != num_constraints_orig:
        error_msg = "Đầu vào ma trận A, b, c hoặc loại ràng buộc không hợp lệ."
        logger.error(error_msg)
        return {'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {}, 'error_message': error_msg, 'parameter_conditions': ""}
    
    if variable_types is None:
        variable_types = ['>=0'] * num_vars_orig 
    elif len(variable_types) != num_vars_orig:
        error_msg = "Độ dài của variable_types phải bằng số lượng biến."
        logger.error(error_msg)
        return {'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {}, 'error_message': error_msg, 'parameter_conditions': ""}

    A_transformed: List[List[float]] = []
    b_transformed: List[float] = []
    constraint_types_transformed: List[str] = []


    for i in range(num_constraints_orig):
        ct = constraint_types[i].strip()
        if not isinstance(b[i], (int, float)):
            error_msg = f"Giá trị b[{i}]='{b[i]}' không phải là số."
            logger.error(error_msg)
            return {'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {}, 'error_message': error_msg, 'parameter_conditions': ""}

        if ct == '=': 
            A_transformed.append(A[i][:])
            b_transformed.append(b[i])
            constraint_types_transformed.append('<=')
            
            A_transformed.append([-coef for coef in A[i]])
            b_transformed.append(-b[i])
            constraint_types_transformed.append('<=')
        elif ct == '>=': 
            A_transformed.append([-coef for coef in A[i]])
            b_transformed.append(-b[i])
            constraint_types_transformed.append('<=')
        elif ct == '<=': 
            A_transformed.append(A[i][:])
            b_transformed.append(b[i])
            constraint_types_transformed.append('<=')
        else:
            error_msg = f"Loại ràng buộc không được hỗ trợ: {ct}"
            logger.error(error_msg)
            return {'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {}, 'error_message': error_msg, 'parameter_conditions': ""}
    
    c_effective = c[:] 
    is_maximization = objective_type.strip().lower().startswith('max')
    if is_maximization:
        c_effective = [-ci for ci in c_effective] 

    parameter_vars_from_min: List[str] = []
    try:
        status_min_problem, z_star_for_min_problem, sol_vals_from_min, steps_min, parameter_vars_from_min = _simplex_min(
            A_transformed, b_transformed, c_effective, 
            constraint_types_transformed, 
            variable_types, 
            "min" 
        )
    except Exception as e:
        logger.error(f"Lỗi trong quá trình giải Simplex: {str(e)}", exc_info=True)
        return {
            'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {},
            'error_message': str(e), 'parameter_conditions': ""
        }

    status_after_solver = status_min_problem 
    
    if status_after_solver == 'Optimal' or status_after_solver == 'Multiple':
        can_attempt_numeric_verification = True
        if status_after_solver == 'Multiple' and parameter_vars_from_min:
            if sol_vals_from_min: # Check if sol_vals_from_min is not empty
                for var_name_orig, sym_expr_val in sol_vals_from_min.items():
                    if hasattr(sym_expr_val, 'free_symbols') and any(str(fs) in parameter_vars_from_min for fs in sym_expr_val.free_symbols):
                        can_attempt_numeric_verification = False 
                        break
            else: # sol_vals_from_min is empty, cannot verify numerically
                can_attempt_numeric_verification = False

        
        if can_attempt_numeric_verification and sol_vals_from_min: # Ensure sol_vals_from_min before iterating
            is_solution_fully_numeric_for_verification = True
            numeric_solution_for_verification: Dict[str, float] = {}
            for i_orig_var in range(num_vars_orig):
                var_name_orig = f"x{i_orig_var+1}"
                sym_expr_val = sol_vals_from_min.get(var_name_orig)
                if sym_expr_val is None or (hasattr(sym_expr_val, 'free_symbols') and sym_expr_val.free_symbols):
                    is_solution_fully_numeric_for_verification = False
                    break
                try:
                    numeric_solution_for_verification[var_name_orig] = float(sym_expr_val.evalf(chop=True))
                except:
                    is_solution_fully_numeric_for_verification = False
                    break
            
            if is_solution_fully_numeric_for_verification:
                solution_verified_feasible = True
                # Placeholder for detailed verification logic
                # for i_constraint in range(num_constraints_orig):
                #     lhs_val = sum(A[i_constraint][j] * numeric_solution_for_verification.get(f"x{j+1}", 0) for j in range(num_vars_orig))
                #     # ... comparison logic ...
                #     if not solution_verified_feasible: break
                if not solution_verified_feasible:
                    logger.warning("Nghiệm số không thỏa mãn ràng buộc gốc. Ghi đè trạng thái.")
                    # status_min_problem = 'Error' # Or 'Infeasible'
                    # sol_vals_from_min.clear()
                    # z_star_for_min_problem = None
    
    z_final_value_for_original_problem: Union[float, str, None] = None 
    if status_min_problem == 'Optimal' or status_min_problem == 'Multiple':
        if z_star_for_min_problem is not None: 
            z_final_value_for_original_problem = -z_star_for_min_problem if is_maximization else z_star_for_min_problem
        else: 
            if status_min_problem == 'Multiple' and steps_min:
                 last_step_title = list(steps_min.keys())[-1]
                 # Ensure 'z' key exists in the last step's tableau data
                 final_tableau_z_data = steps_min[last_step_title]
                 if 'z' in final_tableau_z_data:
                     final_tableau_z_expr_sympy = final_tableau_z_data['z']
                     if final_tableau_z_expr_sympy is not None:
                         if any(str(s) in parameter_vars_from_min for s in final_tableau_z_expr_sympy.free_symbols):
                             z_expr_to_format = -final_tableau_z_expr_sympy if is_maximization else final_tableau_z_expr_sympy
                             z_final_value_for_original_problem = format_expression_for_printing(z_expr_to_format)
                         else: 
                             try:
                                num_z = float(final_tableau_z_expr_sympy.evalf(chop=True))
                                z_final_value_for_original_problem = -num_z if is_maximization else num_z
                             except:
                                z_final_value_for_original_problem = "Lỗi: Z không xác định (tham số)"
                                status_min_problem = 'Error'
                     else: # final_tableau_z_expr_sympy is None
                        z_final_value_for_original_problem = "Lỗi: Z không có trong bảng cuối"
                        status_min_problem = 'Error'
                 else: # 'z' key missing
                    z_final_value_for_original_problem = "Lỗi: Z không tìm thấy trong bảng cuối"
                    status_min_problem = 'Error'

            elif status_min_problem == 'Optimal' : 
                 z_final_value_for_original_problem = "Lỗi: Z tối ưu không xác định"
                 status_min_problem = 'Error'


    elif status_min_problem == 'Unbounded': 
        z_final_value_for_original_problem = float('inf') if is_maximization else float('-inf')
    elif status_min_problem == 'Infeasible':
        z_final_value_for_original_problem = "Không áp dụng (Vô nghiệm)"

    solution_output_formatted: Dict[str, str] = {}
    if (status_min_problem == 'Optimal' or status_min_problem == 'Multiple') and sol_vals_from_min: 
        for var_name, sym_expr in sol_vals_from_min.items(): 
            formatted_expr_str = format_expression_for_printing(sym_expr)
            
            params_in_this_expr = []
            if status_min_problem == 'Multiple' and hasattr(sym_expr, 'free_symbols'):
                for fs_sym in sym_expr.free_symbols:
                    fs_str = str(fs_sym)
                    if fs_str in parameter_vars_from_min: # parameter_vars_from_min are actual tableau parameters
                        params_in_this_expr.append(fs_str)
            
            if params_in_this_expr:
                unique_params_in_expr = sorted(list(set(params_in_this_expr)))
                if isinstance(sym_expr, Symbol) and str(sym_expr) in unique_params_in_expr: # Variable IS a parameter
                    solution_output_formatted[var_name] = f"{formatted_expr_str} (với {str(sym_expr)} >= 0)"
                else: # Variable's expression CONTAINS parameters
                    conditions_for_line = ", ".join([f"{p} >= 0" for p in unique_params_in_expr])
                    solution_output_formatted[var_name] = f"{formatted_expr_str} (tham số: {conditions_for_line})"
            else:
                solution_output_formatted[var_name] = formatted_expr_str
    
    parameter_conditions_str = ""
    if status_min_problem == 'Multiple' and parameter_vars_from_min:
        relevant_parameters = set()
        if sol_vals_from_min:
            for expr in sol_vals_from_min.values():
                if hasattr(expr, 'free_symbols'):
                    for s_sym in expr.free_symbols:
                        s_str = str(s_sym)
                        if s_str in parameter_vars_from_min:
                            relevant_parameters.add(s_str)
        
        if isinstance(z_final_value_for_original_problem, str): 
            try:
                # Attempt to parse the formatted string back to sympy to find free symbols for Z
                # This is a bit indirect; ideally, we'd use the sympy Z expression before formatting.
                # However, z_final_value_for_original_problem might have been negated and formatted.
                # For simplicity, we rely on parameters found in solution variables.
                # A more robust way would be to check free_symbols of the z_expr_to_format if it was symbolic.
                pass # Keep relevant_parameters based on solution variables for now
            except (SyntaxError, TypeError): pass

        if relevant_parameters: 
            conditions = [f"{p} >= 0" for p in sorted(list(relevant_parameters))]
            if conditions:
                parameter_conditions_str = "Điều kiện tham số chung: " + ", ".join(conditions) + "."
        elif not relevant_parameters and parameter_vars_from_min: 
             logger.info("Tham số được xác định trong bảng nhưng không xuất hiện trong biểu thức nghiệm cuối cùng của biến gốc hoặc Z.")


    formatted_steps_output = {}
    for title, tableau_data in steps_min.items():
        fmt_tab = {}
        # Ensure 'z' and other variables are Sympy objects before formatting
        for var, expr_val_maybe_sympy in tableau_data.items():
            if not isinstance(expr_val_maybe_sympy, (Add, Mul, Symbol, Number, type(S.Zero), type(S.One))):
                try:
                    expr_val_sympy = sympify(expr_val_maybe_sympy)
                except (SyntaxError, TypeError, AttributeError):
                    expr_val_sympy = str(expr_val_maybe_sympy) # Fallback to string if cannot sympify
            else:
                expr_val_sympy = expr_val_maybe_sympy
            fmt_tab[var] = format_expression_for_printing(expr_val_sympy)
        formatted_steps_output[title] = fmt_tab
    
    status_map_vn = {
        'Optimal': 'Tối ưu (Optimal)', 
        'Multiple': 'Vô số nghiệm (Multiple Optima)', 
        'Unbounded': 'Không giới nội (Unbounded)',
        'Infeasible': 'Vô nghiệm (Infeasible)', 
        'Processing': 'Đang xử lý (Processing)',
        'Error': 'Lỗi (Error)',
        'MaxIterations': 'Đạt giới hạn vòng lặp (Max Iterations)'
    }
    final_status_str_vn = status_map_vn.get(status_min_problem, status_min_problem)
    
    z_display_value: str
    if isinstance(z_final_value_for_original_problem, str): 
        z_display_value = z_final_value_for_original_problem
    elif z_final_value_for_original_problem is None:
        z_display_value = "N/A"
    else:
        if z_final_value_for_original_problem == float('inf'): z_display_value = "inf"
        elif z_final_value_for_original_problem == float('-inf'): z_display_value = "-inf"
        else: z_display_value = f"{z_final_value_for_original_problem:.2f}"


    return {
        'status': final_status_str_vn,
        'z': z_display_value,
        'solution': solution_output_formatted, 
        'steps': formatted_steps_output,
        'error_message': None,
        'parameter_conditions': parameter_conditions_str
    }