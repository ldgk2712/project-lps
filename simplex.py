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
        except (SyntaxError, TypeError): # Không thể parse nếu nó đã là một chuỗi như "Unbounded"
            return str(expression)

    expression = simplify(expression)
    
    if not expression.free_symbols:
        try:
            if isinstance(expression, Number):
                 # Làm tròn số tới một số chữ số thập phân hợp lý, ví dụ 10, để tránh lỗi float
                 # sau đó mới format về 2 chữ số.
                 num_val = float(expression.evalf(chop=True)) # chop=True loại bỏ số hạng rất nhỏ
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
            coeff_rounded = round(coeff_val, 2) # Làm tròn để hiển thị
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

        if abs(coeff_rounded) < SIMPLEX_TOLERANCE / 100: # Coi như là 0 nếu rất nhỏ
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
    
    decision_vars_orig = sorted([k for k in exprs if k != 'z' and (k.startswith('x') or k.startswith('y')) and not (k.endswith('_p') or k.endswith('_n'))])
    urs_parts = sorted([k for k in exprs if k != 'z' and (k.endswith('_p') or k.endswith('_n'))])
    slack_vars = sorted([k for k in exprs if k != 'z' and k.startswith('w')])
    other_vars = sorted([k for k in exprs if k != 'z' and k not in decision_vars_orig and k not in urs_parts and k not in slack_vars])
    
    ordered_var_names.extend(decision_vars_orig)
    ordered_var_names.extend(urs_parts)
    ordered_var_names.extend(slack_vars)
    ordered_var_names.extend(other_vars) 

    for k_expr in exprs.keys():
        if k_expr not in ordered_var_names:
            ordered_var_names.append(k_expr)

    print(f"{'Biến':<15} | {'Biểu thức':<52}") 
    print(f"{'-' * 15} | {'-' * 52}")

    for var_name in ordered_var_names:
        if var_name in exprs:
            expression_to_format = exprs[var_name]
            if not isinstance(expression_to_format, (Add, Mul, Symbol, Number, type(S.Zero), type(S.One))):
                 expression_to_format = sympify(expression_to_format)
            
            formatted_str = format_expression_for_printing(expression_to_format)
            print(f"{var_name:<15} | {formatted_str:<52}")
            
    if basic_vars or non_basic_vars:
        print(f"{'-' * 70}")
        if basic_vars:
            print(f"Biến cơ bản    : {', '.join(sorted(basic_vars))}")
        if non_basic_vars:
            print(f"Biến không cơ bản: {', '.join(sorted(non_basic_vars))}")
    print(f"{'=' * 70}")


def _simplex_min(A: List[List[float]], b: List[float], c: List[float], 
                 constraint_types: List[str], variable_types: List[str], 
                 objective_type: str) -> Tuple[str, Union[float, None], Dict[str, Any], Dict[str, Any]]:
    """
    Thuật toán Simplex cốt lõi cho bài toán min.
    Hàm này xử lý các biến không bị giới hạn (URS) và phát hiện trường hợp vô số nghiệm.
    LƯU Ý: Đây là một trình giải Simplex cơ bản và có thể không mạnh mẽ bằng các thư viện chuyên dụng
    cho tất cả các trường hợp (ví dụ: suy biến nặng, vấn đề số).
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
        else:  # Default is '>=0'
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
    for j_col in range(len(c_tableau)): # Should iterate up to the number of coeffs in c_tableau
        z_expr += S(c_tableau[j_col]) * all_tableau_symbols[j_col]
    current_tableau['z'] = simplify(z_expr)

    for i_row in range(m): 
        row_expr_for_basic_var = S(b_std[i_row])
        idx_A_col = 0
        for decision_sym in tableau_decision_symbols: 
            # A_tableau[i_row] has coefficients for decision vars, then for slacks
            # We only need the decision var part here for the initial setup.
            if idx_A_col < len(A_tableau[i_row]) - m : # Ensure we are in the decision var coefficient part
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

    while iteration_count < max_iterations:
        iteration_count += 1
        z_row_expr = current_tableau['z']
        entering_var_name: Union[str, None] = None 
        most_negative_coeff_in_z = S.Zero 

        for nb_var_str in non_basic_var_names:
            nb_sym = Symbol(nb_var_str)
            coeff = z_row_expr.coeff(nb_sym) 
            if isinstance(coeff, Number) and float(coeff.evalf()) < float(most_negative_coeff_in_z.evalf() - SIMPLEX_TOLERANCE) : 
                most_negative_coeff_in_z = coeff
                entering_var_name = nb_var_str
        
        if entering_var_name is None or float(most_negative_coeff_in_z.evalf()) >= -SIMPLEX_TOLERANCE:
            _internal_has_alternative_optima_condition = False
            non_basics_with_zero_coeff_in_z = [] 
            for nb_var_str in non_basic_var_names:
                coeff = z_row_expr.coeff(Symbol(nb_var_str))
                if isinstance(coeff, Number) and abs(float(coeff.evalf())) < SIMPLEX_TOLERANCE:
                    _internal_has_alternative_optima_condition = True
                    non_basics_with_zero_coeff_in_z.append(nb_var_str)
            
            if _internal_has_alternative_optima_condition:
                status = 'Multiple' 
                logger.info(f"Điều kiện tableau cho vô số nghiệm được phát hiện. Biến NB có hệ số 0 trong Z: {', '.join(non_basics_with_zero_coeff_in_z)}.")
            else:
                status = 'Optimal'
                logger.info("Tìm thấy nghiệm tối ưu duy nhất cho tableau.")
            break 

        entering_var_symbol = Symbol(entering_var_name) 
        leaving_var_name: Union[str, None] = None 
        min_positive_ratio = float('inf') 
        
        found_candidate_for_leaving = False
        for b_var_str in basic_var_names:
            constraint_expr_for_basic_var = current_tableau[b_var_str] 
            coeff_of_entering_in_basic_var_expr = constraint_expr_for_basic_var.coeff(entering_var_symbol)
            pivot_col_coeff = -coeff_of_entering_in_basic_var_expr # This is A_ij in standard tableau

            if float(pivot_col_coeff.evalf()) > SIMPLEX_TOLERANCE: 
                found_candidate_for_leaving = True
                # current_rhs_of_basic is the constant term of the basic variable's expression
                current_rhs_val = constraint_expr_for_basic_var.subs({Symbol(s): 0 for s in non_basic_var_names if s != entering_var_name})
                current_rhs_val = float(current_rhs_val.as_coeff_Add()[0].evalf())


                if current_rhs_val < -SIMPLEX_TOLERANCE: 
                    # This indicates potential infeasibility if not handled by Dual Simplex or Phase I.
                    # For Primal Simplex, we typically need RHS >= 0 for ratio test.
                    # If we allow negative RHS in ratio test, it can lead to issues or cycling without proper rules.
                    # logger.warning(f"RHS âm ({current_rhs_val:.2f}) cho biến cơ bản {b_var_str} trong ratio test. Điều này có thể dẫn đến vấn đề.")
                    # Standard primal simplex ratio test requires RHS >= 0. If negative, this row isn't a candidate in the standard way.
                    # However, some texts allow ratio test with negative RHS if pivot_col_coeff is also negative (not our case here, as pivot_col_coeff > 0)
                    # For now, if RHS is negative and pivot_col_coeff is positive, this ratio would be negative.
                    # We are looking for MIN_POSITIVE_RATIO.
                     pass # Let the ratio be calculated; negative ratios won't be chosen.

                ratio = current_rhs_val / float(pivot_col_coeff.evalf())

                if ratio >= -SIMPLEX_TOLERANCE: # Allow ratio to be effectively zero
                    if ratio < min_positive_ratio - SIMPLEX_TOLERANCE: # Found a smaller positive ratio
                        min_positive_ratio = ratio
                        leaving_var_name = b_var_str
                    elif abs(ratio - min_positive_ratio) < SIMPLEX_TOLERANCE: # Tie-breaking (Bland's rule by var name)
                        if leaving_var_name is None or b_var_str < leaving_var_name: 
                             min_positive_ratio = ratio # Keep the smaller ratio if it's numerically very close
                             leaving_var_name = b_var_str
        
        if not found_candidate_for_leaving: 
            status = 'Unbounded'
            # This means for the chosen entering_var, all pivot_col_coeffs were <= 0.
            title_unbounded = f"Bước {step+1} (Biến vào: {entering_var_name}, Không có biến ra hợp lệ - Không giới nội)"
            steps_history[title_unbounded] = {k: v.copy() if hasattr(v, 'copy') else v for k,v in current_tableau.items()}
            _print_tableau(title_unbounded, current_tableau, basic_var_names, non_basic_var_names)
            logger.warning(f"Bài toán không giới nội tại bước {step+1} (biến vào {entering_var_name}).")
            break

        if leaving_var_name is None: 
            # This implies found_candidate_for_leaving might have been true,
            # but no valid leaving variable was chosen (e.g., all ratios were negative, or issues with RHS values).
            # This is a strong indicator of infeasibility if the problem was set up correctly.
            status = 'Infeasible' 
            title_issue = f"Bước {step+1} (Biến vào: {entering_var_name}, Không tìm thấy biến ra hợp lệ cho ratio test)"
            steps_history[title_issue] = {k: v.copy() if hasattr(v, 'copy') else v for k,v in current_tableau.items()}
            _print_tableau(title_issue, current_tableau, basic_var_names, non_basic_var_names)
            logger.warning(f"Không tìm thấy biến ra hợp lệ tại bước {step+1} cho biến vào {entering_var_name}. Bài toán có thể vô nghiệm.")
            break
        
        step += 1
        pivot_row_expr = current_tableau[leaving_var_name]
        coeff_entering_in_pivot_row = pivot_row_expr.coeff(entering_var_symbol) 
        
        new_entering_var_value_expr = (Symbol(leaving_var_name) - (pivot_row_expr - coeff_entering_in_pivot_row * entering_var_symbol)) / coeff_entering_in_pivot_row
        
        new_tableau = {entering_var_name: simplify(new_entering_var_value_expr)}

        for var_name_in_tableau, old_expr_in_tableau in current_tableau.items():
            if var_name_in_tableau == leaving_var_name: 
                continue
            # if var_name_in_tableau != entering_var_name: # This check is implicit as entering_var_name is already set
            new_row_expr = old_expr_in_tableau.subs(entering_var_symbol, new_entering_var_value_expr)
            new_tableau[var_name_in_tableau] = simplify(new_row_expr)
        
        current_tableau = new_tableau
        basic_var_names.remove(leaving_var_name)
        basic_var_names.append(entering_var_name)
        non_basic_var_names.remove(entering_var_name)
        non_basic_var_names.append(leaving_var_name)
        
        basic_var_names.sort(); non_basic_var_names.sort()

        title_step_n = f"Bước {step} (Biến vào: {entering_var_name}, Biến ra: {leaving_var_name})"
        current_tableau_for_history = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in current_tableau.items()}
        steps_history[title_step_n] = current_tableau_for_history
        _print_tableau(title_step_n, current_tableau, basic_var_names, non_basic_var_names)

    if iteration_count >= max_iterations:
        logger.warning(f"Đã đạt tối đa {max_iterations} vòng lặp. Dừng lại.")
        status = "MaxIterations"

    z_star_value: Union[float, None] = None
    solution_values: Dict[str, Any] = {} 

    if status == 'Optimal' or status == 'Multiple':
        z_final_expr = current_tableau.get('z', S.Zero)
        subs_all_non_basics_to_zero_for_z_val = {Symbol(nb_str): 0 for nb_str in non_basic_var_names}
        try:
            z_star_value = float(z_final_expr.subs(subs_all_non_basics_to_zero_for_z_val).evalf(chop=True))
        except Exception as e:
            logger.error(f"Không thể tính giá trị Z*: {e}. Biểu thức Z: {z_final_expr}")
            z_star_value = None 
            status = 'Error' # If Z can't be computed, it's an error

        if status != 'Error': # Proceed to get solution values only if Z was computed
            subs_for_solution_expr = {} 
            if status == 'Multiple':
                z_row_expr_for_check = current_tableau['z'] 
                for nb_str in non_basic_var_names:
                    nb_sym = Symbol(nb_str)
                    coeff_in_z = z_row_expr_for_check.coeff(nb_sym)
                    if not (isinstance(coeff_in_z, Number) and abs(float(coeff_in_z.evalf())) < SIMPLEX_TOLERANCE):
                        subs_for_solution_expr[nb_sym] = 0 
            else: 
                for nb_str in non_basic_var_names:
                    subs_for_solution_expr[Symbol(nb_str)] = 0
            
            for info in original_var_info_map:
                original_var_name = info['original_name']
                val_expr_final_for_original_var = S.Zero 
                is_urs_var = info.get('is_urs', False)
                is_transformed_neg_var = info.get('is_transformed_neg', False)

                if is_urs_var:
                    pos_name = info['pos_name']; neg_name = info['neg_name']
                    pos_sym = Symbol(pos_name); neg_sym = Symbol(neg_name)
                    pos_val_or_expr = current_tableau.get(pos_name, pos_sym) if pos_name in basic_var_names else pos_sym
                    neg_val_or_expr = current_tableau.get(neg_name, neg_sym) if neg_name in basic_var_names else neg_sym
                    val_expr_final_for_original_var = (pos_val_or_expr - neg_val_or_expr).subs(subs_for_solution_expr)
                elif is_transformed_neg_var:
                    tableau_name = info['tableau_name']; tab_sym = Symbol(tableau_name)
                    tableau_val_or_expr = current_tableau.get(tableau_name, tab_sym) if tableau_name in basic_var_names else tab_sym
                    val_expr_final_for_original_var = (-tableau_val_or_expr).subs(subs_for_solution_expr)
                else: 
                    tableau_name = info['tableau_name']; tab_sym = Symbol(tableau_name)
                    if tableau_name in basic_var_names:
                        val_expr_final_for_original_var = current_tableau[tableau_name].subs(subs_for_solution_expr)
                    elif tableau_name in non_basic_var_names: # Non-basic, so its value is in subs_for_solution_expr or it's a parameter
                        val_expr_final_for_original_var = subs_for_solution_expr.get(tab_sym, tab_sym) # Default to symbol if it's a parameter
                solution_values[original_var_name] = simplify(val_expr_final_for_original_var)

            if status == 'Multiple':
                is_truly_parametric_for_original_vars = False
                for orig_var_name, sym_expr_val in solution_values.items():
                    if hasattr(sym_expr_val, 'free_symbols') and sym_expr_val.free_symbols:
                        is_truly_parametric_for_original_vars = True; break
                if not is_truly_parametric_for_original_vars:
                    logger.info("Tableau cho thấy vô số nghiệm, nhưng nghiệm cho các biến gốc là duy nhất. Điều chỉnh trạng thái thành 'Tối ưu'.")
                    status = 'Optimal' 
    elif status == 'Unbounded':
        z_star_value = float('-inf') # For min problem, Z goes to -inf.
    # For 'Infeasible', 'Error', 'MaxIterations', z_star_value remains None
    
    return status, z_star_value, solution_values, steps_history


def auto_simplex(
    A: List[List[float]], # Original A matrix
    b: List[float],       # Original b vector
    c: List[float],       # Original c vector
    constraint_types: List[str], # Original constraint types
    objective_type: str = 'max',
    variable_types: List[str] | None = None, # Original variable types
) -> Dict[str, Any]:
    """Hàm chính để giải bài toán quy hoạch tuyến tính bằng Simplex."""
    num_constraints_orig, num_vars_orig = len(A), len(c)
    tol = SIMPLEX_TOLERANCE # Use the defined tolerance

    if not A or not all(len(row) == num_vars_orig for row in A) or \
       len(b) != num_constraints_orig or len(constraint_types) != num_constraints_orig:
        raise ValueError("Đầu vào ma trận A, b, c hoặc loại ràng buộc không hợp lệ.")
    
    if variable_types is None:
        variable_types = ['>=0'] * num_vars_orig 
    elif len(variable_types) != num_vars_orig:
        raise ValueError("Độ dài của variable_types phải bằng số lượng biến.")

    # Standardize constraints for _simplex_min (which expects Ax_tableau + Iw = b_tableau, from Ax <= b)
    # This means all constraints effectively become '<=' for _simplex_min's internal logic.
    A_transformed: List[List[float]] = []
    b_transformed: List[float] = []

    for i in range(num_constraints_orig):
        ct = constraint_types[i].strip()
        if ct == '=':
            # row = b  =>  row <= b AND -row <= -b
            A_transformed.append(A[i][:])
            b_transformed.append(b[i])
            A_transformed.append([-coef for coef in A[i]])
            b_transformed.append(-b[i])
        elif ct == '>=':
            # row >= b => -row <= -b
            A_transformed.append([-coef for coef in A[i]])
            b_transformed.append(-b[i])
        elif ct == '<=':
            A_transformed.append(A[i][:])
            b_transformed.append(b[i])
        else:
            raise ValueError(f"Loại ràng buộc không được hỗ trợ: {ct}")
    
    # c_effective is for the minimization problem solved by _simplex_min
    c_effective = c[:] 
    is_maximization = objective_type.strip().lower().startswith('max')
    if is_maximization:
        c_effective = [-ci for ci in c_effective] 

    status_min_problem: str
    z_star_for_min_problem: Union[float, None]
    sol_vals_from_min: Dict[str, Any]
    steps_min: Dict[str, Dict[str, Any]]

    try:
        # _simplex_min solves Min (c_effective * x_tableau)
        # subject to A_transformed * x_tableau + I*w = b_transformed (implicitly)
        status_min_problem, z_star_for_min_problem, sol_vals_from_min, steps_min = _simplex_min(
            A_transformed, b_transformed, c_effective, 
            ['<='] * len(A_transformed), # All are effectively '<=' for _simplex_min's slack logic
            variable_types, # Original variable types are passed for URS/<=0 transformation
            "min" # _simplex_min always minimizes
        )
    except Exception as e:
        logger.error(f"Lỗi trong quá trình giải Simplex: {str(e)}", exc_info=True)
        return {
            'status': 'Lỗi (Error)', 'z': "N/A", 'solution': {}, 'steps': {},
            'error_message': str(e)
        }

    # --- Verification Step ---
    # Store status from solver before potential override by verification
    status_after_solver = status_min_problem 
    
    if status_after_solver == 'Optimal' or status_after_solver == 'Multiple':
        numeric_solution_for_verification: Dict[str, float] = {}
        is_solution_fully_numeric = True
        has_parameters_in_optimal_solution = False

        for i_orig_var in range(num_vars_orig):
            var_name_orig = f"x{i_orig_var+1}"
            sym_expr_val = sol_vals_from_min.get(var_name_orig)

            if sym_expr_val is None: # Should not happen if solver claims optimal/multiple
                is_solution_fully_numeric = False
                logger.error(f"Lỗi nghiêm trọng: Biến {var_name_orig} không có trong nghiệm dù trạng thái là {status_after_solver}.")
                status_min_problem = 'Error' # Treat as error
                break
            
            if hasattr(sym_expr_val, 'free_symbols') and sym_expr_val.free_symbols:
                if status_after_solver == 'Optimal': # 'Optimal' should not have free symbols
                    has_parameters_in_optimal_solution = True
                # For 'Multiple', parameters are expected, so we don't verify this specific instance numerically.
                # The tableau itself is the source of truth for parametric multiple optima.
                is_solution_fully_numeric = False 
                break 
            
            try:
                numeric_solution_for_verification[var_name_orig] = float(sym_expr_val.evalf(chop=True))
            except (TypeError, AttributeError, ValueError) as e_verif:
                is_solution_fully_numeric = False
                logger.warning(f"Không thể chuyển đổi giá trị nghiệm {sym_expr_val} của {var_name_orig} thành số để xác minh: {e_verif}")
                status_min_problem = 'Error' # If a value can't be numeric, it's an error for verification
                break
        
        if status_min_problem == 'Error': # If error occurred during numeric conversion
            pass # status is already set
        elif has_parameters_in_optimal_solution:
            logger.error(f"Lỗi: Trạng thái 'Tối ưu' nhưng nghiệm cho {var_name_orig} chứa tham số: {sym_expr_val}. Điều này không nhất quán.")
            status_min_problem = 'Error'
        elif is_solution_fully_numeric:
            solution_verified_feasible = True
            # 1. Check variable types against original variable_types
            for i_orig_var in range(num_vars_orig):
                var_name_orig = f"x{i_orig_var+1}"
                val = numeric_solution_for_verification[var_name_orig]
                v_type = variable_types[i_orig_var].strip()
                if v_type == '>=0' and val < -tol: solution_verified_feasible = False
                elif v_type == '<=0' and val > tol: solution_verified_feasible = False
                if not solution_verified_feasible:
                    logger.warning(f"Xác minh thất bại: {var_name_orig}={val:.4f} vi phạm loại biến gốc {v_type}")
                    break
            
            # 2. Check constraints against original A, b, constraint_types
            if solution_verified_feasible:
                for i_constraint in range(num_constraints_orig): # Use original constraints
                    lhs_val = sum(A[i_constraint][j_var] * numeric_solution_for_verification[f"x{j_var+1}"] for j_var in range(num_vars_orig))
                    rhs_val = b[i_constraint]
                    ctype = constraint_types[i_constraint].strip()

                    if ctype == '<=' and lhs_val > rhs_val + tol: solution_verified_feasible = False
                    elif ctype == '>=' and lhs_val < rhs_val - tol: solution_verified_feasible = False
                    elif ctype == '=' and abs(lhs_val - rhs_val) > tol: solution_verified_feasible = False
                    
                    if not solution_verified_feasible:
                        logger.warning(f"Xác minh thất bại: Ràng buộc gốc {i_constraint+1} ({A[i_constraint]} {ctype} {rhs_val}) bị vi phạm. LHS={lhs_val:.4f}")
                        break
            
            if not solution_verified_feasible:
                logger.warning("Nghiệm từ Simplex không thỏa mãn các điều kiện gốc. Ghi đè trạng thái thành 'Vô nghiệm'.")
                status_min_problem = 'Infeasible'
                sol_vals_from_min.clear() # Clear the invalid solution
                z_star_for_min_problem = None # Z value is not relevant for infeasible
    # --- End Verification Step ---

    z_final_value_for_original_problem: Union[float, str, None] = None 
    if status_min_problem == 'Optimal' or status_min_problem == 'Multiple':
        if z_star_for_min_problem is not None:
            z_final_value_for_original_problem = -z_star_for_min_problem if is_maximization else z_star_for_min_problem
        else: # Should not happen if status is Optimal/Multiple and no error occurred
            logger.error(f"Trạng thái {status_min_problem} nhưng z_star_for_min_problem là None.")
            z_final_value_for_original_problem = "Lỗi: Z không xác định" 
            status_min_problem = 'Error'
    elif status_min_problem == 'Unbounded': 
        z_final_value_for_original_problem = float('inf') if is_maximization else float('-inf')
    elif status_min_problem == 'Infeasible':
        z_final_value_for_original_problem = float('-inf') if is_maximization else float('inf')
    # For 'Error', 'MaxIterations', 'Processing', z_final_value_for_original_problem remains None or error string

    solution_output_formatted: Dict[str, str] = {}
    if status_min_problem == 'Optimal' or status_min_problem == 'Multiple': 
        if sol_vals_from_min: # Ensure there are solution values to format
            for var_name, sym_expr in sol_vals_from_min.items(): 
                solution_output_formatted[var_name] = format_expression_for_printing(sym_expr)
        elif status_after_solver in ['Optimal', 'Multiple']: # If original status was Opt/Mult but verification failed
             logger.warning(f"Trạng thái cuối cùng là {status_min_problem} nhưng không có nghiệm để định dạng (có thể do xác minh thất bại).")


    formatted_steps_output = {}
    for title, tableau_data in steps_min.items():
        fmt_tab = {}
        for var, expr_val in tableau_data.items():
            fmt_tab[var] = format_expression_for_printing(expr_val)
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
    if isinstance(z_final_value_for_original_problem, str): # Error string
        z_display_value = z_final_value_for_original_problem
    elif z_final_value_for_original_problem is None:
        # This case should ideally be covered by specific statuses like Infeasible/Unbounded setting inf/-inf
        # Or Error status setting an error string for Z.
        # If it's still None, it implies an unhandled state or MaxIterations/Processing.
        if status_min_problem == 'Infeasible':
            z_display_value = str(float('-inf') if is_maximization else float('inf'))
        elif status_min_problem == 'Unbounded':
            z_display_value = str(float('inf') if is_maximization else float('-inf'))
        else: # MaxIterations, Processing, or an Error that didn't set z_final to a string
            z_display_value = "N/A"
    else: # float, including inf/-inf
        z_display_value = str(z_final_value_for_original_problem)

    return {
        'status': final_status_str_vn,
        'z': z_display_value,
        'solution': solution_output_formatted, 
        'steps': formatted_steps_output,
        'error_message': None # Error message from exception is handled by initial try-except
                           # Specific error messages can be part of 'status' or 'z' if needed.
    }

