# simplex_two_phase.py

import logging
from typing import List, Dict, Tuple, Any, Union, Optional, Set
from sympy import simplify, solve, Symbol, S, sympify, Add, Mul, Number, Expr

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SIMPLEX_TOLERANCE = 1e-9 # Dung sai cho so sánh số thực

# --- Các hàm trợ giúp ---

def format_expression_for_printing(expression: Any) -> str:
    """
    Định dạng biểu thức SymPy để in, làm tròn số đến 2 chữ số thập phân và sắp xếp các số hạng.
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
        if var_part == S.One and term.is_Symbol: var_sym = term
        elif var_part != S.One and var_part.is_Symbol: var_sym = var_part
        elif len(term.free_symbols) == 1:
            var_sym = list(term.free_symbols)[0]
            coeff = term.coeff(var_sym)
        else:
            continue
        if var_sym is not None: var_term_dict[var_sym] = var_term_dict.get(var_sym, S.Zero) + coeff

    sorted_var_symbols = sorted(var_term_dict.keys(), key=lambda s: get_bland_key(str(s)))
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
            if coeff == S.One: var_str_list.append(("+ ", str(var_sym)))
            elif coeff == S.NegativeOne: var_str_list.append(("- ", str(var_sym)))
            else: var_str_list.append(("+ ", f"{str(coeff)}*{var_sym}"))

    if not var_str_list: return const_str
    result_str = const_str if const_str != "0.00" else ""
    for i, (sign, term_s) in enumerate(var_str_list):
        if result_str: result_str += f" {sign}{term_s}"
        else: result_str = term_s if sign == "+ " else f"-{term_s}"
    return result_str if result_str else "0.00"

def get_bland_key(var_obj: Union[str, Symbol]) -> Tuple[int, int, int, str]:
    var_name_str = str(var_obj)
    type_priority, main_index, sub_index = 99, 9999, 0
    base_name = var_name_str
    if var_name_str.endswith('_p'): sub_index, base_name = 1, var_name_str[:-2]
    elif var_name_str.endswith('_n'): sub_index, base_name = 2, var_name_str[:-2]

    if base_name == 'x0': type_priority = -1
    elif base_name.startswith('x') or base_name.startswith('y'): type_priority = 0
    elif base_name.startswith('w') or base_name.startswith('s') or base_name.startswith('u'): type_priority = 1
    elif base_name.startswith('a'): type_priority = 2

    try:
        numeric_part = ''.join(filter(str.isdigit, base_name))
        main_index = int(numeric_part) if numeric_part else (0 if len(base_name) == 1 and type_priority == 0 else 9998)
    except ValueError: pass
    return (type_priority, main_index, sub_index, var_name_str)

def _print_tableau(title: str, exprs: Dict[str, Any],
                   basic_vars: Optional[List[str]] = None,
                   non_basic_vars: Optional[List[str]] = None,
                   objective_var_name: str = 'S') -> None:
    log_output = [f"\n{'=' * 80}", f"{title:^80}", f"{'-' * 80}"]

    ordered_var_names = []
    if objective_var_name in exprs:
        ordered_var_names.append(objective_var_name)

    other_keys = [key for key in exprs.keys() if key != objective_var_name]
    other_keys_sorted = sorted(other_keys, key=get_bland_key)
    ordered_var_names.extend(other_keys_sorted)

    log_output.append(f"{'Biến':<15} | {'Biểu thức':<62}")
    log_output.append(f"{'-' * 15} | {'-' * 62}")
    for var_name in ordered_var_names:
        if var_name in exprs:
            log_output.append(f"{var_name:<15} | {format_expression_for_printing(exprs[var_name]):<62}")

    if basic_vars or non_basic_vars:
        log_output.append(f"{'-' * 80}")
        if basic_vars:
            sorted_display_basics = sorted([bv for bv in basic_vars if bv in exprs], key=get_bland_key)
            log_output.append(f"Biến cơ sở    : {', '.join(sorted_display_basics)}")
        if non_basic_vars:
            sorted_display_non_basics = sorted(non_basic_vars, key=get_bland_key)
            log_output.append(f"Biến không cơ bản: {', '.join(sorted_display_non_basics)}")
    log_output.append(f"{'=' * 80}")
    logger.info("\n".join(log_output))

def _get_ordered_tableau_for_history(tableau_dict: Dict[str, Expr], objective_var_name_param: str) -> List[Tuple[str, str]]:
    ordered_vars = []
    if objective_var_name_param in tableau_dict:
        ordered_vars.append(objective_var_name_param)

    other_keys_list = sorted(
        [k for k in tableau_dict.keys() if k != objective_var_name_param],
        key=get_bland_key
    )
    ordered_vars.extend(other_keys_list)

    return [(var, format_expression_for_printing(tableau_dict[var])) for var in ordered_vars if var in tableau_dict]

def _get_current_solution_values(
    current_tableau: Dict[str, Any], 
    basic_var_names: List[str],
    non_basic_var_names: List[str],
    original_var_info_map: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Tính toán giá trị số của các biến quyết định gốc tại một bước nhất định.
    Hàm này được sao chép từ simplex_bland.py để đảm bảo tính nhất quán.
    """
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
            tableau_name = info['transformed_name']
            solution_values_numeric[original_var_name] = tableau_vars_values.get(tableau_name, 0.0)
        
        elif info.get('is_negated', False):
            tableau_name = info['transformed_name']
            solution_values_numeric[original_var_name] = -tableau_vars_values.get(tableau_name, 0.0)
            
        elif info.get('is_urs', False):
            pos_name = info['p_name']
            neg_name = info['n_name']
            pos_val = tableau_vars_values.get(pos_name, 0.0)
            neg_val = tableau_vars_values.get(neg_name, 0.0)
            solution_values_numeric[original_var_name] = pos_val - neg_val
            
    return solution_values_numeric


# --- Bộ giải Simplex lõi (_simplex_core_solver) ---
def _simplex_core_solver(
    initial_tableau: Dict[str, Expr], initial_basic_vars: List[str], initial_non_basic_vars: List[str],
    objective_var_name: str, phase_name: str,
    original_var_info_map: List[Dict[str, Any]],
    is_maximization_problem: bool = False
) -> Tuple[str, Optional[float], Dict[str, Expr], Dict[str, Expr], List[str], List[str], Dict[str, Dict[str, Any]]]:

    current_tableau = {}
    for k, v_expr in initial_tableau.items():
        if isinstance(v_expr, Number):
            current_tableau[k] = v_expr
        elif hasattr(v_expr, 'copy'):
            try:
                current_tableau[k] = v_expr.copy()
            except TypeError:
                current_tableau[k] = sympify(v_expr)
        else:
            current_tableau[k] = v_expr

    basic_vars = initial_basic_vars[:]
    non_basic_vars = initial_non_basic_vars[:]
    
    steps_history: Dict[str, Dict[str, Any]] = {}
    step_counter = 0
    title_step_0 = f'{phase_name} - Bước {step_counter} (Bảng khởi tạo)'
    
    coords_step_0_dict = _get_current_solution_values(current_tableau, basic_vars, non_basic_vars, original_var_info_map)
    coords_step_0_list = [coords_step_0_dict.get('x1', 0.0), coords_step_0_dict.get('x2', 0.0)]

    steps_history[title_step_0] = {
        'tableau': _get_ordered_tableau_for_history(current_tableau, objective_var_name),
        'coords': coords_step_0_list
    }
    _print_tableau(title_step_0, current_tableau, basic_vars, non_basic_vars, objective_var_name)

    max_iterations, iteration_count = 100, 0
    while iteration_count < max_iterations:
        iteration_count += 1
        basic_var_symbols = [Symbol(s) for s in basic_vars]
        non_basic_var_symbols = [Symbol(s) for s in non_basic_vars]
        if objective_var_name not in current_tableau:
            logger.error(f"{phase_name}: Biến mục tiêu '{objective_var_name}' không tìm thấy trong bảng hiện tại. Dừng lại.")
            return 'Error', None, {}, current_tableau, basic_vars, non_basic_vars, steps_history

        objective_row_expr = current_tableau[objective_var_name]
        is_degenerate = any(
            b_var_str in current_tableau and \
            current_tableau[b_var_str].subs({Symbol(nb_s): S.Zero for nb_s in non_basic_vars}).is_Number and \
            abs(float(current_tableau[b_var_str].subs({Symbol(nb_s): S.Zero for nb_s in non_basic_vars}).evalf())) < SIMPLEX_TOLERANCE
            for b_var_str in basic_vars
        )

        if is_degenerate: logger.info(f"{phase_name}: Phát hiện suy biến.")

        entering_var_sym: Optional[Symbol] = None
        best_coeff_for_entering = S.Zero + SIMPLEX_TOLERANCE
        sorted_non_basic_candidates = sorted(non_basic_var_symbols, key=get_bland_key)

        if is_degenerate:
            for nb_sym_candidate in sorted_non_basic_candidates:
                coeff_in_obj = objective_row_expr.coeff(nb_sym_candidate)
                if coeff_in_obj.is_Number and float(coeff_in_obj.evalf()) < -SIMPLEX_TOLERANCE:
                    entering_var_sym = nb_sym_candidate
                    break
        else:
            for nb_sym_candidate in sorted_non_basic_candidates:
                coeff_in_obj = objective_row_expr.coeff(nb_sym_candidate)
                if coeff_in_obj.is_Number and float(coeff_in_obj.evalf()) < float(best_coeff_for_entering.evalf() - SIMPLEX_TOLERANCE):
                    best_coeff_for_entering, entering_var_sym = coeff_in_obj, nb_sym_candidate

        if entering_var_sym is None:
            status = 'Optimal'
            logger.info(f"{phase_name}: {status}.")
            break

        min_positive_ratio, potential_leaving_vars, found_positive_pivot_candidate = float('inf'), [], False
        for b_var_sym_candidate in basic_var_symbols:
            b_var_str_candidate = str(b_var_sym_candidate)
            if b_var_str_candidate not in current_tableau: continue
            constraint_expr = current_tableau[b_var_str_candidate]
            pivot_column_coeff_in_row = -constraint_expr.coeff(entering_var_sym)
            if pivot_column_coeff_in_row.is_Number and float(pivot_column_coeff_in_row.evalf()) > SIMPLEX_TOLERANCE:
                found_positive_pivot_candidate = True
                rhs_val_expr = constraint_expr.subs({Symbol(nb_s): S.Zero for nb_s in non_basic_vars})
                if rhs_val_expr.is_Number:
                    rhs_val = float(rhs_val_expr.evalf())
                    if rhs_val >= -SIMPLEX_TOLERANCE:
                        actual_rhs_for_ratio = max(0, rhs_val)
                        pivot_val_float = float(pivot_column_coeff_in_row.evalf())
                        ratio = actual_rhs_for_ratio / pivot_val_float if pivot_val_float != 0 else float('inf')
                        potential_leaving_vars.append((ratio, b_var_sym_candidate))

        if not found_positive_pivot_candidate:
            status = 'Unbounded'
            coords_unbounded_dict = _get_current_solution_values(current_tableau, basic_vars, non_basic_vars, original_var_info_map)
            coords_unbounded_list = [coords_unbounded_dict.get('x1', 0.0), coords_unbounded_dict.get('x2', 0.0)]
            title_unbounded = f'{phase_name} - Bước {step_counter+1} (Vào: {str(entering_var_sym)}, Không giới nội)'
            steps_history[title_unbounded] = {
                'tableau': _get_ordered_tableau_for_history(current_tableau, objective_var_name),
                'coords': coords_unbounded_list
            }
            break

        min_ratio_val = min(r for r, v_sym in potential_leaving_vars)
        tied_leaving_vars = [v_sym for r, v_sym in potential_leaving_vars if abs(r - min_ratio_val) < SIMPLEX_TOLERANCE]
        tied_leaving_vars.sort(key=get_bland_key)
        leaving_var_sym = tied_leaving_vars[0]

        step_counter += 1
        
        pivot_row_expr_old = current_tableau[str(leaving_var_sym)]
        coeff_entering_in_pivot_row_expr = pivot_row_expr_old.coeff(entering_var_sym)

        P_rest = simplify(pivot_row_expr_old - coeff_entering_in_pivot_row_expr * entering_var_sym)
        substitution_expr_for_entering_var = simplify((leaving_var_sym - P_rest) / coeff_entering_in_pivot_row_expr)

        new_tableau_temp = {str(entering_var_sym): substitution_expr_for_entering_var}
        for var_name_iter, old_expr_iter in current_tableau.items():
            if var_name_iter != str(leaving_var_sym):
                new_tableau_temp[var_name_iter] = simplify(old_expr_iter.subs(entering_var_sym, substitution_expr_for_entering_var))

        current_tableau = new_tableau_temp

        basic_vars.remove(str(leaving_var_sym)); basic_vars.append(str(entering_var_sym))
        non_basic_vars.remove(str(entering_var_sym)); non_basic_vars.append(str(leaving_var_sym))

        title_step_n = f'{phase_name} - Bước {step_counter} (Vào: {str(entering_var_sym)}, Ra: {str(leaving_var_sym)})'
        coords_step_n_dict = _get_current_solution_values(current_tableau, basic_vars, non_basic_vars, original_var_info_map)
        coords_step_n_list = [coords_step_n_dict.get('x1', 0.0), coords_step_n_dict.get('x2', 0.0)]
        steps_history[title_step_n] = {
            'tableau': _get_ordered_tableau_for_history(current_tableau, objective_var_name),
            'coords': coords_step_n_list
        }
        _print_tableau(title_step_n, current_tableau, basic_vars, non_basic_vars, objective_var_name)

    if iteration_count >= max_iterations: status = "MaxIterations"

    final_objective_value: Optional[float] = None
    final_solution_expressions: Dict[str, Expr] = {}
    if status == 'Optimal' or status == 'Multiple Optima':
        obj_expr_at_opt = current_tableau.get(objective_var_name, S.Zero)
        subs_final_nb_to_zero = {Symbol(nb_s): S.Zero for nb_s in non_basic_vars}
        try: final_objective_value = float(obj_expr_at_opt.subs(subs_final_nb_to_zero).evalf(chop=True))
        except Exception: status = 'Error'

        for b_var_str in basic_vars:
            final_solution_expressions[b_var_str] = current_tableau.get(b_var_str, S.Zero)
        for nb_var_str in non_basic_vars:
            final_solution_expressions[nb_var_str] = S.Zero

    elif status == 'Unbounded': final_objective_value = float('-inf')
    return status, final_objective_value, final_solution_expressions, current_tableau, basic_vars, non_basic_vars, steps_history

# --- Hàm Simplex Hai Pha Chính ---
def simplex_two_phase(
    A_orig: List[List[float]], b_orig: List[float], c_orig: List[float],
    constraint_types_orig: List[str], variable_types_orig: List[str],
    objective_type_orig: str = 'max'
) -> Dict[str, Any]:
    
    A_processed, b_processed, constraint_types_processed = [], [], []
    has_equality = any(ct.strip() == '=' for ct in constraint_types_orig)
    if has_equality:
        for i, constraint_type in enumerate(constraint_types_orig):
            if constraint_type.strip() == '=':
                A_processed.append(A_orig[i]); b_processed.append(b_orig[i]); constraint_types_processed.append('<=')
                A_processed.append(A_orig[i]); b_processed.append(b_orig[i]); constraint_types_processed.append('>=')
            else:
                A_processed.append(A_orig[i]); b_processed.append(b_orig[i]); constraint_types_processed.append(constraint_type)
    else:
        A_processed, b_processed, constraint_types_processed = A_orig, b_orig, constraint_types_orig
    
    num_original_vars = len(c_orig)
    num_constraints = len(b_processed)

    original_var_info_map: List[Dict[str, Any]] = []
    A_eff_cols: List[List[Expr]] = [[] for _ in range(num_constraints)]
    c_eff_sympy_list: List[Expr] = []
    decision_vars_transformed_symbols: List[Symbol] = []

    for i in range(num_original_vars):
        original_name = f"x{i+1}"
        var_type = variable_types_orig[i]
        var_info = {'original_name': original_name, 'original_idx': i, 'type': var_type}
        if var_type == '<=0':
            y_name = f"y{i + 1}"; y_sym = Symbol(y_name)
            for r in range(num_constraints): A_eff_cols[r].append(-S(A_processed[r][i]))
            c_eff_sympy_list.append(-S(c_orig[i])); decision_vars_transformed_symbols.append(y_sym)
            var_info.update({'transformed_name': y_name, 'transformed_symbol': y_sym, 'is_negated': True})
        elif var_type == 'URS':
            p_name = f"{original_name}_p"; p_sym = Symbol(p_name)
            n_name = f"{original_name}_n"; n_sym = Symbol(n_name)
            for r in range(num_constraints): A_eff_cols[r].extend([S(A_processed[r][i]), -S(A_processed[r][i])])
            c_eff_sympy_list.extend([S(c_orig[i]), -S(c_orig[i])]); decision_vars_transformed_symbols.extend([p_sym, n_sym])
            var_info.update({'p_name': p_name, 'p_symbol': p_sym, 'n_name': n_name, 'n_symbol': n_sym, 'is_urs': True})
        else: # '>=0'
            x_sym = Symbol(original_name)
            for r in range(num_constraints): A_eff_cols[r].append(S(A_processed[r][i]))
            c_eff_sympy_list.append(S(c_orig[i])); decision_vars_transformed_symbols.append(x_sym)
            var_info.update({'transformed_name': original_name, 'transformed_symbol': x_sym, 'is_standard': True})
        original_var_info_map.append(var_info)

    A_eff = [[A_eff_cols[r][c] for c in range(len(decision_vars_transformed_symbols))] for r in range(num_constraints)]
    b_eff_exprs = [S(val) for val in b_processed]

    combined_steps: Dict[str, Dict[str, Any]] = {}
    x0_sym = Symbol('x0')
    tableau_p1: Dict[str, Expr] = {}
    basic_vars_p1_initial, non_basic_vars_p1_initial = [], [str(s) for s in decision_vars_transformed_symbols] + [str(x0_sym)]

    for i in range(num_constraints):
        w_name = f"w{i+1}"; w_sym = Symbol(w_name)
        basic_vars_p1_initial.append(w_name)
        sum_Ax_term_expr = sum(A_eff[i][k] * decision_vars_transformed_symbols[k] for k in range(len(decision_vars_transformed_symbols)))
        if constraint_types_processed[i] == '>=':
            tableau_p1[w_name] = simplify(x0_sym - b_eff_exprs[i] + sum_Ax_term_expr)
        elif constraint_types_processed[i] == '<=':
            tableau_p1[w_name] = simplify(x0_sym + b_eff_exprs[i] - sum_Ax_term_expr)
    
    S_sym = Symbol('S'); tableau_p1[str(S_sym)] = x0_sym
    
    min_const_in_wj, leaving_var_w_name_for_x0_pivot = S.Infinity, None
    for w_name in basic_vars_p1_initial:
        const_part = tableau_p1[w_name].subs(x0_sym, S.Zero).subs({s: S.Zero for s in decision_vars_transformed_symbols})
        if const_part.is_Number and float(const_part.evalf()) < -SIMPLEX_TOLERANCE and const_part < min_const_in_wj:
            min_const_in_wj, leaving_var_w_name_for_x0_pivot = const_part, w_name

    tableau_p1_pre_pivoted, basic_vars_p1_pre_pivoted, non_basic_vars_p1_pre_pivoted = tableau_p1.copy(), basic_vars_p1_initial[:], non_basic_vars_p1_initial[:]
    
    if leaving_var_w_name_for_x0_pivot:
        expr_for_x0 = Symbol(leaving_var_w_name_for_x0_pivot) - simplify(tableau_p1[leaving_var_w_name_for_x0_pivot].subs(x0_sym, S.Zero))
        for var_name, expr in tableau_p1_pre_pivoted.items():
            tableau_p1_pre_pivoted[var_name] = simplify(expr.subs(x0_sym, expr_for_x0))
        basic_vars_p1_pre_pivoted.remove(leaving_var_w_name_for_x0_pivot); basic_vars_p1_pre_pivoted.append(str(x0_sym))
        non_basic_vars_p1_pre_pivoted.remove(str(x0_sym)); non_basic_vars_p1_pre_pivoted.append(leaving_var_w_name_for_x0_pivot)

    status_p1, min_S_value, _, final_tableau_p1, \
    final_basic_vars_p1, final_non_basic_vars_p1, steps_p1_from_core = _simplex_core_solver(
        tableau_p1_pre_pivoted, basic_vars_p1_pre_pivoted, non_basic_vars_p1_pre_pivoted,
        str(S_sym), "Phase 1", original_var_info_map
    )
    combined_steps.update(steps_p1_from_core)

    if status_p1 != 'Optimal' or (min_S_value is not None and abs(min_S_value) > SIMPLEX_TOLERANCE):
        return {'status': 'Vô nghiệm (Infeasible)', 'z': "N/A", 'solution': {}, 'steps': combined_steps, 'error_message': f"Pha 1 kết thúc với min S = {min_S_value}"}

    tableau_p2: Dict[str, Expr] = {}
    basic_vars_p2 = [bv for bv in final_basic_vars_p1 if bv != str(S_sym) and bv != str(x0_sym)]
    non_basic_vars_p2 = [nb for nb in final_non_basic_vars_p1 if nb != str(S_sym) and nb != str(x0_sym)]

    for b_var in basic_vars_p2:
        if b_var in final_tableau_p1:
            tableau_p2[b_var] = simplify(final_tableau_p1[b_var].subs(x0_sym, S.Zero))

    is_max_problem = objective_type_orig.lower() == 'max'
    z_original_expr = sum(c_eff_sympy_list[i] * decision_vars_transformed_symbols[i] for i in range(len(c_eff_sympy_list)))
    z_expr_for_solver = -z_original_expr if is_max_problem else z_original_expr
    
    subs_for_z = {Symbol(b_var): tableau_p2[b_var] for b_var in basic_vars_p2 if Symbol(b_var) in z_expr_for_solver.free_symbols and b_var in tableau_p2}
    tableau_p2['z_obj'] = simplify(z_expr_for_solver.subs(subs_for_z))

    status_p2, z_solver_value, sol_exprs_p2, _, _, _, steps_p2_from_core = _simplex_core_solver(
        tableau_p2, basic_vars_p2, non_basic_vars_p2,
        'z_obj', "Phase 2", original_var_info_map
    )
    combined_steps.update(steps_p2_from_core)

    final_z_value: Union[float, str, None] = "N/A"
    if z_solver_value is not None:
        if status_p2 == 'Unbounded': final_z_value = float('inf') if is_max_problem else float('-inf')
        elif status_p2 == 'Optimal': final_z_value = -z_solver_value if is_max_problem else z_solver_value
    
    solution_final_orig_vars: Dict[str, Any] = {}
    if status_p2 == 'Optimal' and sol_exprs_p2:
        for var_info in original_var_info_map:
            orig_name = var_info['original_name']
            val_expr: Expr = S.Zero
            if var_info['type'] == '>=0':
                val_expr = sol_exprs_p2.get(var_info['transformed_name'], S.Zero)
            elif var_info['type'] == '<=0':
                val_expr = -sol_exprs_p2.get(var_info['transformed_name'], S.Zero)
            elif var_info['type'] == 'URS':
                p_val = sol_exprs_p2.get(var_info['p_name'], S.Zero)
                n_val = sol_exprs_p2.get(var_info['n_name'], S.Zero)
                val_expr = p_val - n_val
            solution_final_orig_vars[orig_name] = format_expression_for_printing(val_expr)

    status_map_vn = {'Optimal': 'Tối ưu (Optimal)', 'Unbounded': 'Không giới nội (Unbounded)', 'Infeasible': 'Vô nghiệm (Infeasible)', 'Error': 'Lỗi (Error)', 'MaxIterations': 'Đạt giới hạn vòng lặp (Max Iterations)'}
    final_status = status_map_vn.get(status_p2, status_p2) if status_p1 == 'Optimal' and (min_S_value is None or abs(min_S_value) < SIMPLEX_TOLERANCE) else status_map_vn.get(status_p1, status_p1)

    z_display = f"{final_z_value:.2f}" if isinstance(final_z_value, (float, int)) else str(final_z_value)
    
    return {'status': final_status, 'z': z_display, 'solution': solution_final_orig_vars, 'steps': combined_steps, 'error_message': None}
