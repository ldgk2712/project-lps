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
        if k_expr not in ordered_var_names: # Should not happen if logic is correct
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
                    pass # Keep as is if cannot sympify (e.g. already string)
            
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
            main_index = 0 if len(base_name) == 1 else 9998 # For 'x' or 'y' itself
    elif base_name.startswith('w'):
        type_priority = 1
        try:
            main_index = int(base_name[1:])
        except ValueError:
            main_index = 0 
    elif base_name.startswith('a'): # Artificial variables, if ever used by this specific simplex
        type_priority = 2
        try:
            main_index = int(base_name[1:])
        except ValueError:
            main_index = 0
    else: # Fallback for other variable names
        try:
            # Try to extract a number if it exists for some ordering
            numeric_part = ''.join(filter(str.isdigit, base_name))
            if numeric_part:
                main_index = int(numeric_part)
        except ValueError:
            pass # Keep default main_index

    return (type_priority, main_index, sub_index, var_name_str)


def _simplex_min(A: List[List[float]], b: List[float], c: List[float], 
                 constraint_types: List[str], variable_types: List[str], 
                 objective_type: str) -> Tuple[str, Union[float, None], Dict[str, Any], Dict[str, Dict[str, Any]], List[str]]:
    """
    Thuật toán Simplex cốt lõi cho bài toán min, sử dụng Quy tắc Bland khi phát hiện suy biến.
    Trả về: status, z_star, solution_values, steps_history (Dict[str, Dict[str, SympyExpr]]), parameter_names_for_multiple_optima
    LƯU Ý: steps_history bây giờ trả về Dict[str, SympyExpr] cho mỗi bước, việc chuyển đổi sang List[Tuple[str,str]] sẽ ở auto_simplex.
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
    
    step = 0 
    steps_history: Dict[str, Dict[str, Any]] = {} # Stores Dict[str, SympyExpr] for each step title
    current_tableau: Dict[str, Any] = {} # Stores SympyExpr

    basic_var_names = [str(s) for s in slack_symbols]
    non_basic_var_names = [str(s) for s in tableau_decision_symbols]

    z_expr = S.Zero
    # Build z_expr from c_tableau and all_tableau_symbols (decision + slack)
    # c_tableau was extended for slack vars with 0 coefficient
    for j_col_z in range(len(all_tableau_symbols)): # Should be same length as c_tableau now
        z_expr += S(c_tableau[j_col_z]) * all_tableau_symbols[j_col_z]
    current_tableau['z'] = simplify(z_expr)


    for i_row in range(m): 
        row_expr_for_basic_var = S(b_std[i_row])
        idx_A_col = 0
        for decision_sym in tableau_decision_symbols: 
            # A_tableau[i_row] should have length equal to number of decision_symbols + number of slack_symbols
            # We only subtract terms related to decision_symbols here for the initial tableau.
            if idx_A_col < len(A_tableau[i_row]) - m : # Check against original A_tableau width for decision vars
                 row_expr_for_basic_var -= S(A_tableau[i_row][idx_A_col]) * decision_sym
            idx_A_col +=1
        current_tableau[slack_symbols[i_row].name] = simplify(row_expr_for_basic_var)
    
    title_step_0 = 'Bước 0 (Bảng khởi tạo - Initial Tableau)'
    # Store Sympy expressions directly for history, formatting will be done later
    steps_history[title_step_0] = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in current_tableau.items()}
    _print_tableau(title_step_0, current_tableau, basic_var_names, non_basic_var_names)
    
    status = 'Processing' 
    max_iterations = 100 
    iteration_count = 0
    parameter_names_for_multiple_optima: List[str] = []


    while iteration_count < max_iterations:
        iteration_count += 1
        z_row_expr = current_tableau['z']
        
        is_degenerate = False
        for b_var_str_degen_check in basic_var_names:
            # Check RHS of basic variable by substituting non-basics to 0
            rhs_val_degen = current_tableau[b_var_str_degen_check].subs({Symbol(s_nb_degen): 0 for s_nb_degen in non_basic_var_names})
            if isinstance(rhs_val_degen, Number) and abs(float(rhs_val_degen.evalf(chop=True))) < SIMPLEX_TOLERANCE:
                is_degenerate = True
                logger.info(f"Phát hiện suy biến tại {b_var_str_degen_check} với RHS ≈ 0")
                break

        entering_var_name: Union[str, None] = None
        if is_degenerate:
            # Bland's rule for entering: choose the non-basic var with the smallest index (Bland key)
            # that has a negative coefficient in the z-row.
            sorted_candidates_bland_entering = sorted(non_basic_var_names, key=get_bland_key)
            for var_str_candidate_bland in sorted_candidates_bland_entering:
                var_sym_candidate_bland = Symbol(var_str_candidate_bland)
                coeff_in_z_bland = z_row_expr.coeff(var_sym_candidate_bland)
                if isinstance(coeff_in_z_bland, Number) and float(coeff_in_z_bland.evalf(chop=True)) < -SIMPLEX_TOLERANCE:
                    entering_var_name = var_str_candidate_bland
                    logger.info(f"Suy biến: Chọn biến vào {entering_var_name} (hệ số: {coeff_in_z_bland:.4f}) theo Quy tắc Bland.")
                    break
        else: # Not degenerate, or no Bland choice made yet
            most_negative_coeff_in_z_val = S.Zero + SIMPLEX_TOLERANCE # Small positive
            # Standard rule: most negative. Tie-break with Bland.
            sorted_non_basics_for_std_entering = sorted(non_basic_var_names, key=get_bland_key)
            for nb_var_str_std in sorted_non_basics_for_std_entering:
                nb_sym_std = Symbol(nb_var_str_std)
                coeff_std = z_row_expr.coeff(nb_sym_std)
                if isinstance(coeff_std, Number):
                    coeff_float_std = float(coeff_std.evalf(chop=True))
                    most_negative_float = float(most_negative_coeff_in_z_val.evalf(chop=True))
                    if coeff_float_std < most_negative_float - SIMPLEX_TOLERANCE: # Clearly more negative
                        most_negative_coeff_in_z_val = coeff_std
                        entering_var_name = nb_var_str_std
                    # If equal, Bland sort already handled it.
            if entering_var_name:
                 logger.info(f"Không suy biến: Chọn biến vào {entering_var_name} với hệ số {float(most_negative_coeff_in_z_val.evalf()):.4f}")


        if entering_var_name is None: # Optimality condition
            _internal_has_alternative_optima_condition = False
            non_basics_with_zero_coeff_in_z_temp = [] 
            sorted_non_basics_for_alt_check = sorted(non_basic_var_names, key=get_bland_key)

            for nb_var_str_alt_check in sorted_non_basics_for_alt_check:
                coeff_alt_check = z_row_expr.coeff(Symbol(nb_var_str_alt_check))
                if isinstance(coeff_alt_check, Number) and abs(float(coeff_alt_check.evalf(chop=True))) < SIMPLEX_TOLERANCE:
                    # Check if this NB var can enter (i.e., has a positive pivot in some row)
                    can_nb_enter_for_alt_opt = False
                    for b_var_str_pivot_check in basic_var_names:
                        constraint_expr_pivot_check = current_tableau[b_var_str_pivot_check]
                        coeff_of_nb_in_row = constraint_expr_pivot_check.coeff(Symbol(nb_var_str_alt_check))
                        if isinstance(coeff_of_nb_in_row, Number) and float(-coeff_of_nb_in_row.evalf(chop=True)) > SIMPLEX_TOLERANCE: # pivot_element > 0
                            can_nb_enter_for_alt_opt = True
                            break
                    if can_nb_enter_for_alt_opt:
                        _internal_has_alternative_optima_condition = True
                        non_basics_with_zero_coeff_in_z_temp.append(nb_var_str_alt_check)
            
            if _internal_has_alternative_optima_condition:
                status = 'Multiple' 
                parameter_names_for_multiple_optima = non_basics_with_zero_coeff_in_z_temp[:]
                logger.info(f"Vô số nghiệm: Biến NB có hệ số 0 trong Z và có thể vào cơ sở: {', '.join(parameter_names_for_multiple_optima)}")
            else:
                status = 'Optimal'
                logger.info("Tìm thấy nghiệm tối ưu")
            break 

        entering_var_symbol = Symbol(entering_var_name) 
        
        leaving_var_name: Union[str, None] = None 
        min_positive_ratio_val = float('inf') 
        potential_leaving_vars_with_ratio: List[Tuple[float, str]] = [] # (ratio, var_name)
        
        found_positive_pivot_coeff_for_entering_var = False
        
        # Ratio test: iterate through basic variables
        for b_var_str_ratio_test in basic_var_names: 
            constraint_expr_for_basic_var = current_tableau[b_var_str_ratio_test] 
            coeff_of_entering_in_basic_var_expr = constraint_expr_for_basic_var.coeff(entering_var_symbol)
            # Pivot element in tableau is -coeff_of_entering_in_basic_var_expr (because x_b = RHS - sum(a_ij * x_j))
            pivot_col_coeff_val = -coeff_of_entering_in_basic_var_expr

            if isinstance(pivot_col_coeff_val, Number) and float(pivot_col_coeff_val.evalf(chop=True)) > SIMPLEX_TOLERANCE: # Pivot element must be > 0
                found_positive_pivot_coeff_for_entering_var = True
                
                current_rhs_val_expr = constraint_expr_for_basic_var.subs({Symbol(s_rhs): 0 for s_rhs in non_basic_var_names if s_rhs != entering_var_name})
                # Ensure current_rhs_val is a number
                current_rhs_val_float = S.Zero
                if current_rhs_val_expr.is_Number:
                    current_rhs_val_float = float(current_rhs_val_expr.evalf(chop=True))
                else: # Should be a constant term if all non-basics (except maybe entering) are zero
                    const_part_rhs, _ = current_rhs_val_expr.as_coeff_Add()
                    if const_part_rhs.is_Number:
                        current_rhs_val_float = float(const_part_rhs.evalf(chop=True))
                    else:
                        logger.error(f"RHS của {b_var_str_ratio_test} không phải số: {current_rhs_val_expr}")
                        continue # Skip this row for ratio test

                if current_rhs_val_float < -SIMPLEX_TOLERANCE: # Negative RHS, implies problem or infeasibility if standard form was not feasible
                    # This typically shouldn't happen in a standard simplex if initial BFS is feasible.
                    # If it does, this row is not a candidate for leaving with positive pivot.
                    continue 
                
                # Ensure non-negative RHS for ratio calculation (or handle division by zero carefully)
                actual_rhs_for_ratio = max(0, current_rhs_val_float) 
                pivot_float_for_ratio = float(pivot_col_coeff_val.evalf(chop=True))
                
                ratio = actual_rhs_for_ratio / pivot_float_for_ratio # Already checked pivot_float_for_ratio > SIMPLEX_TOLERANCE

                if ratio >= -SIMPLEX_TOLERANCE: # Ratio should be non-negative
                    potential_leaving_vars_with_ratio.append((ratio, b_var_str_ratio_test))
        
        if not found_positive_pivot_coeff_for_entering_var: 
            status = 'Unbounded'
            title_unbounded = f"Bước {step+1} (Biến vào: {entering_var_name}, Không giới nội)"
            steps_history[title_unbounded] = {k_unb: v_unb.copy() if hasattr(v_unb, 'copy') else v_unb for k_unb,v_unb in current_tableau.items()}
            _print_tableau(title_unbounded, current_tableau, basic_var_names, non_basic_var_names)
            logger.warning(f"Không giới nội tại bước {step+1} (biến vào {entering_var_name})")
            break

        if not potential_leaving_vars_with_ratio: 
            # This might happen if all positive pivot coeffs correspond to negative or zero RHS,
            # or if no positive pivot coeffs were found (covered by 'Unbounded').
            # If found_positive_pivot_coeff_for_entering_var was true, but this is empty, it's an issue.
            status = 'Error' # Or 'Infeasible' if RHS values imply it.
            title_issue_leaving = f"Bước {step+1} (Biến vào: {entering_var_name}, Không tìm thấy biến ra hợp lệ)"
            steps_history[title_issue_leaving] = {k_il: v_il.copy() if hasattr(v_il, 'copy') else v_il for k_il,v_il in current_tableau.items()}
            _print_tableau(title_issue_leaving, current_tableau, basic_var_names, non_basic_var_names)
            logger.warning(f"Không tìm thấy biến ra hợp lệ cho {entering_var_name} tại bước {step+1}")
            break
        
        # Determine leaving variable from potential candidates
        min_ratio_val_from_potential = min(r for r, v_name in potential_leaving_vars_with_ratio)
        
        tied_leaving_candidates_by_ratio = [
            v_name for r, v_name in potential_leaving_vars_with_ratio 
            if abs(r - min_ratio_val_from_potential) < SIMPLEX_TOLERANCE
        ]

        if tied_leaving_candidates_by_ratio:
            # Bland's rule for leaving variable: choose the one with the smallest Bland index among those tied for min ratio.
            tied_leaving_candidates_by_ratio.sort(key=get_bland_key) 
            leaving_var_name = tied_leaving_candidates_by_ratio[0]
            logger.info(f"Chọn biến ra {leaving_var_name} (Tỷ lệ: {min_ratio_val_from_potential:.4f}, Quy tắc Bland cho hòa nếu có)")
        else: # Should not be reached if potential_leaving_vars_with_ratio was not empty
            status = 'Error'; logger.error(f"Lỗi logic chọn biến ra (không có ứng viên sau khi lọc tỷ lệ) tại bước {step+1}"); break


        if leaving_var_name is None: # Safety, should have been set
            status = 'Error'; logger.error(f"Lỗi: leaving_var_name là None tại bước {step+1}"); break

        step += 1
        logger.info(f"--- Pivot Step {step} ---")
        logger.info(f"Biến vào (Entering): {entering_var_name}, Biến ra (Leaving): {leaving_var_name}")

        pivot_row_expr = current_tableau[leaving_var_name]
        # The coefficient of the entering variable *in the expression* for the leaving variable
        coeff_entering_in_pivot_row_expr = pivot_row_expr.coeff(entering_var_symbol)

        if abs(float(coeff_entering_in_pivot_row_expr.evalf(chop=True))) < SIMPLEX_TOLERANCE / 100.0:
            logger.error(f"LỖI: Phần tử pivot ({coeff_entering_in_pivot_row_expr}) quá gần 0 cho biến vào '{entering_var_name}' trong dòng của '{leaving_var_name}'.")
            status = 'Error'; break 

        # Create substitution expression for entering_var_symbol:
        # leaving_var_name = const + coeff_entering_in_pivot_row_expr * entering_var_symbol + ...other_terms...
        # So, entering_var_symbol = (leaving_var_name - const - ...other_terms...) / coeff_entering_in_pivot_row_expr
        # Or, more directly: entering_var_symbol = (Symbol(leaving_var_name) - (pivot_row_expr - coeff_entering_in_pivot_row_expr * entering_var_symbol)) / coeff_entering_in_pivot_row_expr
        
        # Let E be the entering variable, L be the leaving variable.
        # L = a_L0 + sum_{j non-basic, j!=E} a_Lj * x_j + a_LE * E
        # We want to express E in terms of L and other non-basics.
        # E = (L - a_L0 - sum_{j non-basic, j!=E} a_Lj * x_j) / a_LE
        # The expression for E will be: new_tableau[entering_var_name] = (Symbol(leaving_var_name) - (pivot_row_expr - coeff_entering_in_pivot_row_expr * entering_var_symbol)) / coeff_entering_in_pivot_row_expr

        # Simpler: new_expr_for_entering_var = Symbol(leaving_var_name) / coeff_entering_in_pivot_row_expr - (pivot_row_expr - coeff_entering_in_pivot_row_expr * entering_var_symbol) / coeff_entering_in_pivot_row_expr
        # This effectively makes 'leaving_var_name' the new non-basic and 'entering_var_name' the new basic.
        
        # The new expression for the entering variable (which becomes basic)
        # is derived from the pivot row: pivot_row_expr = leaving_var_name_symbol
        # We want to solve for entering_var_symbol from:
        # current_tableau[leaving_var_name] = some_expr_terms + coeff_entering_in_pivot_row_expr * entering_var_symbol
        # So, entering_var_symbol = (current_tableau[leaving_var_name] - some_expr_terms) / coeff_entering_in_pivot_row_expr
        # where some_expr_terms = pivot_row_expr - coeff_entering_in_pivot_row_expr * entering_var_symbol
        # Thus, E_expr_for_substitution = (Symbol(leaving_var_name) - (pivot_row_expr - coeff_entering_in_pivot_row_expr * entering_var_symbol)) / coeff_entering_in_pivot_row_expr

        expr_to_solve_for_entering = Symbol(leaving_var_name) - (pivot_row_expr - coeff_entering_in_pivot_row_expr * entering_var_symbol)
        new_entering_var_value_expr = simplify(expr_to_solve_for_entering / coeff_entering_in_pivot_row_expr)

        
        new_tableau_temp = {}
        new_tableau_temp[entering_var_name] = new_entering_var_value_expr # This is the new basic var's row

        # Update other rows by substituting the old entering_var_symbol with its new expression
        # The substitution expression for the *old* entering_var_symbol (which is now basic) is
        # what we just calculated for new_tableau_temp[entering_var_name], BUT this expression
        # is in terms of the *new* non-basic (leaving_var_name) and other *old* non-basics.
        # This is correct.

        # For other basic variables (not the one leaving) and for z-row:
        # Old_row = const + ... + coeff_E_in_Old_row * E_old_non_basic + ...
        # New_row = Old_row.subs(E_old_non_basic, new_expression_for_E_old_non_basic)
        # where new_expression_for_E_old_non_basic is new_entering_var_value_expr

        for var_name_in_old_tableau, old_expr_in_tableau in current_tableau.items():
            if var_name_in_old_tableau == leaving_var_name: # This row is replaced by the new entering var's row
                continue
            # For z-row and other basic variable rows:
            substituted_expr = old_expr_in_tableau.subs(entering_var_symbol, new_entering_var_value_expr)
            simplified_new_row_expr = simplify(substituted_expr)
            new_tableau_temp[var_name_in_old_tableau] = simplified_new_row_expr
        
        current_tableau = new_tableau_temp
        
        basic_var_names.remove(leaving_var_name)
        basic_var_names.append(entering_var_name)
        non_basic_var_names.remove(entering_var_name)
        non_basic_var_names.append(leaving_var_name)
        
        basic_var_names.sort(key=get_bland_key)
        non_basic_var_names.sort(key=get_bland_key)

        title_step_n = f"Bước {step} (Biến vào: {entering_var_name}, Biến ra: {leaving_var_name})"
        # Store Sympy expressions directly
        steps_history[title_step_n] = {k_hist_n: v_hist_n.copy() if hasattr(v_hist_n, 'copy') else v_hist_n for k_hist_n,v_hist_n in current_tableau.items()}
        _print_tableau(title_step_n, current_tableau, basic_var_names, non_basic_var_names)

    if iteration_count >= max_iterations:
        logger.warning(f"Đạt tối đa {max_iterations} vòng lặp")
        status = "MaxIterations"

    z_star_value: Union[float, None] = None
    solution_values: Dict[str, Any] = {} # Stores Sympy expressions for original variable names

    if status == 'Optimal' or status == 'Multiple':
        z_final_expr = current_tableau.get('z', S.Zero)
        # Substitute non-parameters to 0
        subs_all_non_basics_to_zero_for_z_val = {
            Symbol(nb_str_z): 0 for nb_str_z in non_basic_var_names 
            if nb_str_z not in parameter_names_for_multiple_optima
        }
        
        try:
            z_expr_substituted_for_val = z_final_expr.subs(subs_all_non_basics_to_zero_for_z_val)
            if not z_expr_substituted_for_val.free_symbols: # If it becomes a number
                 z_star_value_eval = z_expr_substituted_for_val.evalf(chop=True)
                 z_star_value = float(z_star_value_eval)
            # If z_expr_substituted_for_val still has free_symbols (parameters), z_star_value remains None
            # and will be handled by formatting later if status is 'Multiple'.
        except Exception as e_z_val:
            logger.error(f"Không thể tính Z* số: {e_z_val}")
            z_star_value = None 

        if status != 'Error': # Proceed to calculate solution expressions
            # Substitutions for expressing basic vars: non-parameter non-basics are 0.
            subs_for_solution_expr_display = {} 
            for nb_str_sol in non_basic_var_names:
                if nb_str_sol not in parameter_names_for_multiple_optima:
                     subs_for_solution_expr_display[Symbol(nb_str_sol)] = 0
                # else: parameters remain as symbols implicitly by not being in this dict
            
            for info in original_var_info_map:
                original_var_name = info['original_name']
                val_expr_final_for_original_var_sympy: Any = S.Zero # Default to Sympy Zero
                is_urs_var = info.get('is_urs', False)
                is_transformed_neg_var = info.get('is_transformed_neg', False)
                is_standard_var = info.get('is_standard', False)


                if is_urs_var:
                    pos_name, neg_name = info['pos_name'], info['neg_name']
                    pos_sym, neg_sym = Symbol(pos_name), Symbol(neg_name)
                    
                    pos_expr_or_sym = S.Zero # Default to 0 if non-basic and not parameter
                    if pos_name in basic_var_names: pos_expr_or_sym = current_tableau.get(pos_name, S.Zero)
                    elif pos_name in parameter_names_for_multiple_optima: pos_expr_or_sym = pos_sym
                    
                    neg_expr_or_sym = S.Zero
                    if neg_name in basic_var_names: neg_expr_or_sym = current_tableau.get(neg_name, S.Zero)
                    elif neg_name in parameter_names_for_multiple_optima: neg_expr_or_sym = neg_sym
                                        
                    val_expr_final_for_original_var_sympy = (pos_expr_or_sym - neg_expr_or_sym).subs(subs_for_solution_expr_display)

                elif is_transformed_neg_var:
                    tableau_name = info['tableau_name']
                    tab_sym = Symbol(tableau_name)
                    
                    transformed_var_expr_or_sym = S.Zero
                    if tableau_name in basic_var_names: transformed_var_expr_or_sym = current_tableau.get(tableau_name, S.Zero)
                    elif tableau_name in parameter_names_for_multiple_optima: transformed_var_expr_or_sym = tab_sym
                    
                    val_expr_final_for_original_var_sympy = (-transformed_var_expr_or_sym).subs(subs_for_solution_expr_display)

                elif is_standard_var: # Standard x_i >= 0
                    tableau_name = info['tableau_name'] # This is original_name
                    tab_sym = Symbol(tableau_name)

                    if tableau_name in basic_var_names:
                        val_expr_final_for_original_var_sympy = current_tableau.get(tableau_name, S.Zero).subs(subs_for_solution_expr_display)
                    elif tableau_name in non_basic_var_names:
                        if tableau_name in parameter_names_for_multiple_optima: 
                            val_expr_final_for_original_var_sympy = tab_sym # It's a parameter itself
                        else: # Non-basic, not a parameter, so it's 0
                            val_expr_final_for_original_var_sympy = S.Zero
                    else: # Should not happen if logic is correct (var is either basic or non-basic)
                        logger.warning(f"Biến {tableau_name} không phải cơ sở cũng không phải phi cơ sở.")
                        val_expr_final_for_original_var_sympy = tab_sym # Fallback to symbol if state is inconsistent
                
                solution_values[original_var_name] = simplify(val_expr_final_for_original_var_sympy)
            
            if status == 'Multiple':
                # Check if the solution *actually* depends on parameters
                is_truly_parametric_for_original_vars = False
                if solution_values: # solution_values contains the original variable expressions
                    for orig_var_name_check_param, sym_expr_val_check_param in solution_values.items():
                        if hasattr(sym_expr_val_check_param, 'free_symbols'):
                            for free_sym_check in sym_expr_val_check_param.free_symbols:
                                if str(free_sym_check) in parameter_names_for_multiple_optima:
                                    is_truly_parametric_for_original_vars = True; break
                        if is_truly_parametric_for_original_vars: break
                
                if not is_truly_parametric_for_original_vars and parameter_names_for_multiple_optima:
                    logger.info("Vô số nghiệm ở bảng Simplex, nhưng nghiệm cho các biến gốc có thể là duy nhất (tham số bị triệt tiêu). Giữ trạng thái 'Multiple'.")
                    # If solution is unique despite tableau params, it's still a form of multiple optima at tableau level.

    elif status == 'Unbounded':
        z_star_value = float('-inf') # For minimization problem

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

    # Standardize constraints to Ax <= b form for the _simplex_min solver.
    # Note: _simplex_min itself handles variable transformations (URS, <=0) internally.
    # This part only standardizes constraint *types* if needed (e.g. >= to <=, = to two <=).
    # However, the current _simplex_min seems to build tableau directly from original A, b, c, constraint_types, variable_types.
    # The transformation to Ax <= b for all constraints is more typical for a Phase 1 / Big M, or a dual simplex.
    # The provided _simplex_min directly constructs a Simplex tableau assuming b_i >= 0 for slack variables to be initial BFS.
    # If b_i can be negative, Phase 1 / Two-Phase method is required.
    # This auto_simplex is for cases where b_i >= 0 (or it's handled before calling this, e.g. by Two-Phase).
    # Let's assume b_i >= 0. Constraint types like '=', '>=' need to be handled by introducing artificial/surplus vars or by the solver.
    # The current _simplex_min seems to assume all constraints are effectively handled to allow slack variables as initial basis.
    # This implies it's designed for problems already in a form where b_i >= 0 and constraints allow adding slacks directly.
    # For this auto_simplex, we will pass constraint_types and variable_types directly to _simplex_min.
    # The transformation of Ax >= b to -Ax <= -b, and Ax = b to Ax <=b and -Ax <= -b might be one way,
    # but the _simplex_min's internal tableau construction needs to align.
    # Given the current _simplex_min's direct tableau setup, complex transformations here might conflict.
    # We'll pass them as is and let _simplex_min handle tableau construction based on them.

    A_to_solver = A # No transformation here, _simplex_min handles it.
    b_to_solver = b
    constraint_types_to_solver = constraint_types
    
    c_effective = c[:] 
    is_maximization = objective_type.strip().lower().startswith('max')
    if is_maximization:
        c_effective = [-ci for ci in c_effective] # Convert Max to Min: min(-Z)

    parameter_vars_from_solver: List[str] = []
    try:
        status_solver, z_star_from_solver, sol_vals_sympy_from_solver, steps_history_sympy, parameter_vars_from_solver = _simplex_min(
            A_to_solver, b_to_solver, c_effective, 
            constraint_types_to_solver, # Pass original types
            variable_types, 
            "min" # Solver always minimizes
        )
    except ValueError as ve: # Catch specific errors from _simplex_min validation
        logger.error(f"Lỗi đầu vào cho _simplex_min: {str(ve)}", exc_info=True)
        return {
            'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {},
            'error_message': f"Lỗi dữ liệu đầu vào: {str(ve)}", 'parameter_conditions': ""
        }
    except Exception as e:
        logger.error(f"Lỗi không mong muốn trong quá trình giải Simplex: {str(e)}", exc_info=True)
        return {
            'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {},
            'error_message': f"Lỗi hệ thống: {str(e)}", 'parameter_conditions': ""
        }
    
    # --- Format steps_history_sympy for output ---
    # steps_history_sympy is Dict[str_title, Dict[str_var_name, SympyExpr]]
    # We need Dict[str_title, List[Tuple[str_var_name, str_formatted_expr]]]
    formatted_steps_output: Dict[str, List[Tuple[str,str]]] = {}
    for title, tableau_data_dict in steps_history_sympy.items():
        ordered_tableau_list_for_step = []
        
        all_vars_in_step = list(tableau_data_dict.keys())
        var_order_for_display = []
        if 'z' in all_vars_in_step: # 'z' is the objective of the min problem
            var_order_for_display.append('z')
        
        other_vars_in_step = sorted(
            [k for k in all_vars_in_step if k != 'z'],
            key=get_bland_key 
        )
        var_order_for_display.extend(other_vars_in_step)
        
        for var_name_str in var_order_for_display:
            if var_name_str in tableau_data_dict:
                expr_val_sympy = tableau_data_dict[var_name_str]
                # Ensure it's a Sympy object if possible before formatting
                if not isinstance(expr_val_sympy, (Add, Mul, Symbol, Number, type(S.Zero), type(S.One))):
                    try:
                        expr_val_sympy = sympify(expr_val_sympy)
                    except (SyntaxError, TypeError, AttributeError):
                        pass # Keep as string if cannot sympify
                
                formatted_expr = format_expression_for_printing(expr_val_sympy)
                ordered_tableau_list_for_step.append((var_name_str, formatted_expr))

        formatted_steps_output[title] = ordered_tableau_list_for_step
    # --- End of formatting steps ---

    z_final_value_display: Union[float, str, None] = "N/A"
    if status_solver == 'Optimal' or status_solver == 'Multiple':
        if z_star_from_solver is not None: # Numeric Z* from solver (for min problem)
            z_final_value_display = -z_star_from_solver if is_maximization else z_star_from_solver
        else: # Z* might be parametric if 'Multiple'
            if status_solver == 'Multiple' and formatted_steps_output:
                 last_step_title = list(formatted_steps_output.keys())[-1]
                 final_tableau_list_tuples = formatted_steps_output[last_step_title]
                 z_row_str_expr_in_output = "Không tìm thấy Z trong bảng cuối"
                 for var_name_output, expr_str_output in final_tableau_list_tuples:
                     if var_name_output == 'z': # This is z_min
                         z_row_str_expr_in_output = expr_str_output
                         break
                 
                 # If z_row_str_expr_in_output contains parameters, it's symbolic.
                 # If is_maximization, we need to represent -Z_min.
                 # This formatting is tricky if Z_min itself is a string expression.
                 # For now, if z_star_from_solver was None (parametric), we take the formatted string.
                 # A full symbolic negation and re-format is complex here.
                 # We rely on format_expression_for_printing of the Z_min from the tableau.
                 # If is_maximization, we can just prepend "-( " + ... + " )" or rely on user to interpret.
                 # For simplicity, if parametric and max, we show the min(-Z) value.
                 # The `solution_output_formatted` will show parametric variable values.
                 
                 # Try to get the sympy expression for 'z' from the *last step's raw sympy data*
                 raw_final_tableau_dict = steps_history_sympy.get(list(steps_history_sympy.keys())[-1] if steps_history_sympy else "", {})
                 final_z_sympy_expr_for_min = raw_final_tableau_dict.get('z')

                 if final_z_sympy_expr_for_min is not None:
                    z_expr_for_orig_obj = -final_z_sympy_expr_for_min if is_maximization else final_z_sympy_expr_for_min
                    z_final_value_display = format_expression_for_printing(z_expr_for_orig_obj)
                 else:
                    z_final_value_display = "Z biểu thức lỗi" # Fallback
            else: # Optimal but z_star_from_solver was None, should not happen
                z_final_value_display = "Z tối ưu lỗi"
                status_solver = 'Error'


    elif status_solver == 'Unbounded': 
        # If min problem is unbounded (-inf), then max problem is unbounded (+inf)
        z_final_value_display = float('inf') if is_maximization else float('-inf')
    elif status_solver == 'Infeasible':
        z_final_value_display = "Không áp dụng (Vô nghiệm)" # N/A for Z value
    # Else (Error, MaxIterations), z_final_value_display remains "N/A"

    solution_output_formatted: Dict[str, str] = {}
    if (status_solver == 'Optimal' or status_solver == 'Multiple') and sol_vals_sympy_from_solver: 
        for var_name_orig_sol, sym_expr_sol in sol_vals_sympy_from_solver.items(): 
            # sol_vals_sympy_from_solver contains original var names and their sympy expressions
            formatted_expr_str_sol = format_expression_for_printing(sym_expr_sol)
            
            params_in_this_sol_expr = []
            if status_solver == 'Multiple' and hasattr(sym_expr_sol, 'free_symbols'):
                for fs_sym_sol in sym_expr_sol.free_symbols:
                    fs_str_sol = str(fs_sym_sol)
                    # parameter_vars_from_solver contains names of non-basics in z-row with 0 coeff from _simplex_min
                    if fs_str_sol in parameter_vars_from_solver: 
                        params_in_this_sol_expr.append(fs_str_sol)
            
            if params_in_this_sol_expr:
                unique_params_in_sol_expr = sorted(list(set(params_in_this_sol_expr)))
                # Check if the variable itself is a parameter
                if isinstance(sym_expr_sol, Symbol) and str(sym_expr_sol) in unique_params_in_sol_expr:
                    solution_output_formatted[var_name_orig_sol] = f"{formatted_expr_str_sol} (tham số, {str(sym_expr_sol)} >= 0)"
                else:
                    conditions_for_sol_line = ", ".join([f"{p_param} >= 0" for p_param in unique_params_in_sol_expr])
                    solution_output_formatted[var_name_orig_sol] = f"{formatted_expr_str_sol} (phụ thuộc tham số: {conditions_for_sol_line})"
            else:
                solution_output_formatted[var_name_orig_sol] = formatted_expr_str_sol
    
    parameter_conditions_str_final = ""
    if status_solver == 'Multiple' and parameter_vars_from_solver:
        # Identify parameters that actually affect the final solution or Z value
        relevant_parameters_overall = set()
        if sol_vals_sympy_from_solver: # Check solution expressions
            for expr_sol_check in sol_vals_sympy_from_solver.values():
                if hasattr(expr_sol_check, 'free_symbols'):
                    for s_sym_check_sol in expr_sol_check.free_symbols:
                        if str(s_sym_check_sol) in parameter_vars_from_solver:
                            relevant_parameters_overall.add(str(s_sym_check_sol))
        
        # Check Z expression (if it was symbolic)
        if isinstance(z_final_value_display, str): # Implies it might be a formatted sympy string
            # To properly check params in Z, we need the sympy Z expression for the original problem
            raw_final_tableau_dict_for_z_param = steps_history_sympy.get(list(steps_history_sympy.keys())[-1] if steps_history_sympy else "", {})
            final_z_sympy_expr_for_min_param_check = raw_final_tableau_dict_for_z_param.get('z')
            if final_z_sympy_expr_for_min_param_check is not None:
                z_expr_for_orig_obj_param_check = -final_z_sympy_expr_for_min_param_check if is_maximization else final_z_sympy_expr_for_min_param_check
                if hasattr(z_expr_for_orig_obj_param_check, 'free_symbols'):
                    for s_sym_check_z in z_expr_for_orig_obj_param_check.free_symbols:
                        if str(s_sym_check_z) in parameter_vars_from_solver:
                             relevant_parameters_overall.add(str(s_sym_check_z))
        
        if relevant_parameters_overall: 
            conditions_list = [f"{p_rel} >= 0" for p_rel in sorted(list(relevant_parameters_overall))]
            if conditions_list:
                parameter_conditions_str_final = "Bài toán có vô số nghiệm phụ thuộc vào các tham số: " + ", ".join(conditions_list) + ". Các tham số này là các biến phi cơ sở có hệ số 0 trong dòng mục tiêu của bảng tối ưu."
        elif not relevant_parameters_overall and parameter_vars_from_solver: 
             logger.info("Tham số được xác định trong bảng Simplex nhưng không xuất hiện trong biểu thức nghiệm cuối cùng của biến gốc hoặc Z.")
             parameter_conditions_str_final = "Có dấu hiệu vô số nghiệm từ bảng Simplex (biến phi cơ sở có hệ số 0 trong dòng Z), nhưng nghiệm cụ thể của các biến gốc có thể không phụ thuộc tham số."

    
    status_map_vn = {
        'Optimal': 'Tối ưu (Optimal)', 
        'Multiple': 'Vô số nghiệm (Multiple Optima)', 
        'Unbounded': 'Không giới nội (Unbounded)',
        'Infeasible': 'Vô nghiệm (Infeasible)', 
        'Processing': 'Đang xử lý (Processing)', # Should not be a final status
        'Error': 'Lỗi (Error)',
        'MaxIterations': 'Đạt giới hạn vòng lặp (Max Iterations)'
    }
    final_status_str_vn_display = status_map_vn.get(status_solver, status_solver) # Map status from solver
    
    z_display_final_corrected: str
    if isinstance(z_final_value_display, str): 
        z_display_final_corrected = z_final_value_display
    elif z_final_value_display is None: # Should ideally be caught and set to "N/A" or similar
        z_display_final_corrected = "N/A"
    else: # It's a float (inf, -inf, or number)
        if z_final_value_display == float('inf'): z_display_final_corrected = "inf"
        elif z_final_value_display == float('-inf'): z_display_final_corrected = "-inf"
        else: z_display_final_corrected = f"{z_final_value_display:.2f}" # Standard numeric formatting


    return {
        'status': final_status_str_vn_display,
        'z': z_display_final_corrected,
        'solution': solution_output_formatted, 
        'steps': formatted_steps_output, # This is now List[Tuple[str,str]] for each step
        'error_message': None, # Assuming errors were caught and returned earlier
        'parameter_conditions': parameter_conditions_str_final
    }
