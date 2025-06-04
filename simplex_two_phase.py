# simplex_two_phase_x0_method.py

import logging
from typing import List, Dict, Tuple, Any, Union, Optional, Set
from sympy import simplify, solve, Symbol, S, sympify, Add, Mul, Number, Expr

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SIMPLEX_TOLERANCE = 1e-9 # Dung sai cho so sánh số thực

# --- Các hàm trợ giúp (format_expression_for_printing, get_bland_key, _print_tableau) ---

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
            logger.debug(f"Skipping unusual term in formatter: {term}")
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
    # YÊU CẦU: Đẩy hàng S (hoặc biến mục tiêu) lên đầu.
    # Logic hiện tại đã thực hiện điều này:
    # Biến mục tiêu (objective_var_name, ví dụ 'S' trong Pha 1) được thêm vào danh sách đầu tiên.
    if objective_var_name in exprs:
        ordered_var_names.append(objective_var_name)

    # Lấy tất cả các khóa khác (không phải là hàm mục tiêu) từ bảng
    other_keys = [key for key in exprs.keys() if key != objective_var_name]

    # Sắp xếp các khóa còn lại theo khóa Bland
    other_keys_sorted = sorted(other_keys, key=get_bland_key)

    # Thêm các khóa đã sắp xếp vào danh sách hiển thị (sau biến mục tiêu)
    ordered_var_names.extend(other_keys_sorted)

    log_output.append(f"{'Biến':<15} | {'Biểu thức':<62}")
    log_output.append(f"{'-' * 15} | {'-' * 62}")
    for var_name in ordered_var_names: # In theo thứ tự đã xác định
        if var_name in exprs: # Đảm bảo chỉ in các biến có trong bảng hiện tại
            log_output.append(f"{var_name:<15} | {format_expression_for_printing(exprs[var_name]):<62}")

    if basic_vars or non_basic_vars:
        log_output.append(f"{'-' * 80}")
        if basic_vars:
            # Sắp xếp danh sách biến cơ bản để hiển thị nhất quán
            sorted_display_basics = sorted([bv for bv in basic_vars if bv in exprs], key=get_bland_key)
            log_output.append(f"Biến cơ bản    : {', '.join(sorted_display_basics)}")
        if non_basic_vars:
            # Sắp xếp danh sách biến không cơ bản để hiển thị nhất quán
            sorted_display_non_basics = sorted(non_basic_vars, key=get_bland_key)
            log_output.append(f"Biến không cơ bản: {', '.join(sorted_display_non_basics)}")
    log_output.append(f"{'=' * 80}")
    logger.info("\n".join(log_output))

def _get_ordered_tableau_for_history(tableau_dict: Dict[str, Expr], objective_var_name_param: str) -> List[Tuple[str, str]]:
    """
    Tạo một danh sách các (tên_biến, biểu_thức_đã_định_dạng) cho lịch sử các bước.
    Dòng của biến mục tiêu (objective_var_name_param) sẽ được đặt ở đầu danh sách.
    Các dòng khác được sắp xếp theo khóa Bland.
    """
    ordered_vars = []
    if objective_var_name_param in tableau_dict:
        ordered_vars.append(objective_var_name_param)

    other_keys_list = sorted(
        [k for k in tableau_dict.keys() if k != objective_var_name_param],
        key=get_bland_key
    )
    ordered_vars.extend(other_keys_list)

    return [(var, format_expression_for_printing(tableau_dict[var])) for var in ordered_vars if var in tableau_dict]

# --- Bộ giải Simplex lõi (_simplex_core_solver) ---
def _simplex_core_solver(
    initial_tableau: Dict[str, Expr], initial_basic_vars: List[str], initial_non_basic_vars: List[str],
    objective_var_name: str, phase_name: str,
    is_maximization_problem: bool = False
) -> Tuple[str, Optional[float], Dict[str, Expr], Dict[str, Expr], List[str], List[str], Dict[str, List[Tuple[str, str]]]]: # Thay đổi kiểu trả về cho steps_history

    current_tableau = {}
    for k, v_expr in initial_tableau.items():
        if isinstance(v_expr, Number):
            current_tableau[k] = v_expr
        elif hasattr(v_expr, 'copy'):
            try:
                current_tableau[k] = v_expr.copy()
            except TypeError as e:
                logger.warning(f"TypeError during {k}.copy() for expr {v_expr} (type {type(v_expr)}). Error: {e}. Using sympify as fallback.")
                current_tableau[k] = sympify(v_expr)
        else:
            current_tableau[k] = v_expr

    basic_vars = initial_basic_vars[:]
    non_basic_vars = initial_non_basic_vars[:]
    basic_var_symbols = [Symbol(s) for s in basic_vars]
    non_basic_var_symbols = [Symbol(s) for s in non_basic_vars]

    steps_history: Dict[str, List[Tuple[str, str]]] = {} # Thay đổi kiểu của steps_history
    step_counter = 0
    title_step_0 = f'{phase_name} - Bước {step_counter} (Bảng khởi tạo)'
    # Lưu trữ bảng cho history với thứ tự dòng mục tiêu lên đầu
    steps_history[title_step_0] = _get_ordered_tableau_for_history(current_tableau, objective_var_name)
    _print_tableau(title_step_0, current_tableau, basic_vars, non_basic_vars, objective_var_name)

    max_iterations, iteration_count = 100, 0
    while iteration_count < max_iterations:
        iteration_count += 1
        # Kiểm tra xem objective_var_name có trong current_tableau không trước khi truy cập
        if objective_var_name not in current_tableau:
            logger.error(f"{phase_name}: Biến mục tiêu '{objective_var_name}' không tìm thấy trong bảng hiện tại. Dừng lại.")
            # Có thể trả về trạng thái lỗi ở đây nếu cần
            return 'Error', None, {}, current_tableau, basic_vars, non_basic_vars, steps_history


        objective_row_expr = current_tableau[objective_var_name]
        is_degenerate = any(
            current_tableau[b_var_str].subs({Symbol(nb_s): S.Zero for nb_s in non_basic_vars}).is_Number and \
            abs(float(current_tableau[b_var_str].subs({Symbol(nb_s): S.Zero for nb_s in non_basic_vars}).evalf())) < SIMPLEX_TOLERANCE
            for b_var_str in basic_vars if b_var_str in current_tableau # Thêm kiểm tra b_var_str in current_tableau
        )
        if is_degenerate: logger.info(f"{phase_name}: Phát hiện suy biến.")

        entering_var_sym: Optional[Symbol] = None
        best_coeff_for_entering = S.Zero + SIMPLEX_TOLERANCE
        sorted_non_basic_candidates = sorted(non_basic_var_symbols, key=get_bland_key)

        if is_degenerate:
            for nb_sym_candidate in sorted_non_basic_candidates:
                coeff_in_obj = objective_row_expr.coeff(nb_sym_candidate)
                if coeff_in_obj.is_Number and float(coeff_in_obj.evalf()) < -SIMPLEX_TOLERANCE:
                    entering_var_sym, best_coeff_for_entering = nb_sym_candidate, coeff_in_obj
                    logger.info(f"{phase_name} (Suy biến): Chọn biến vào {str(entering_var_sym)} (hệ số: {format_expression_for_printing(coeff_in_obj)}).")
                    break
        else:
            for nb_sym_candidate in sorted_non_basic_candidates:
                coeff_in_obj = objective_row_expr.coeff(nb_sym_candidate)
                if coeff_in_obj.is_Number and float(coeff_in_obj.evalf()) < float(best_coeff_for_entering.evalf() - SIMPLEX_TOLERANCE):
                    best_coeff_for_entering, entering_var_sym = coeff_in_obj, nb_sym_candidate
            if entering_var_sym: logger.info(f"{phase_name}: Chọn biến vào {str(entering_var_sym)} (hệ số: {format_expression_for_printing(best_coeff_for_entering)}).")

        if entering_var_sym is None:
            status, has_alternative_optima = 'Optimal', False
            for nb_sym_check in non_basic_var_symbols:
                obj_coeff = objective_row_expr.coeff(nb_sym_check)
                if obj_coeff.is_Number and abs(float(obj_coeff.evalf())) < SIMPLEX_TOLERANCE:
                    if any(
                        b_var_str_alt in current_tableau and # Thêm kiểm tra
                        -current_tableau[b_var_str_alt].coeff(nb_sym_check).is_Number and \
                        float(-current_tableau[b_var_str_alt].coeff(nb_sym_check).evalf()) > SIMPLEX_TOLERANCE
                        for b_var_str_alt in basic_vars
                    ):
                        has_alternative_optima = True
                        logger.info(f"{phase_name}: Có thể có nghiệm thay thế (biến {str(nb_sym_check)}).")
                        break
            status = 'Multiple Optima' if has_alternative_optima else 'Optimal'
            logger.info(f"{phase_name}: {status}.")
            break

        min_positive_ratio, potential_leaving_vars, found_positive_pivot_candidate = float('inf'), [], False
        for b_var_sym_candidate in basic_var_symbols:
            b_var_str_candidate = str(b_var_sym_candidate)
            if b_var_str_candidate not in current_tableau: continue # Bỏ qua nếu biến cơ bản không có trong bảng (có thể xảy ra nếu có lỗi trước đó)
            constraint_expr = current_tableau[b_var_str_candidate]
            pivot_column_coeff_in_row = -constraint_expr.coeff(entering_var_sym)
            if pivot_column_coeff_in_row.is_Number and float(pivot_column_coeff_in_row.evalf()) > SIMPLEX_TOLERANCE:
                found_positive_pivot_candidate = True
                rhs_val_expr = constraint_expr.subs({Symbol(nb_s): S.Zero for nb_s in non_basic_vars})
                if rhs_val_expr.is_Number:
                    rhs_val = float(rhs_val_expr.evalf())
                    if rhs_val < -SIMPLEX_TOLERANCE:
                        logger.debug(f"{phase_name}: RHS âm ({rhs_val:.4f}) cho {b_var_str_candidate}. Bỏ qua tỷ lệ."); continue
                    actual_rhs_for_ratio = max(0, rhs_val)
                    pivot_val_float = float(pivot_column_coeff_in_row.evalf())
                    ratio = actual_rhs_for_ratio / pivot_val_float if pivot_val_float != 0 else float('inf')
                    if ratio >= -SIMPLEX_TOLERANCE: potential_leaving_vars.append((ratio, b_var_sym_candidate))
                else: logger.error(f"{phase_name}: RHS của {b_var_str_candidate} không phải số: {rhs_val_expr}")

        if not found_positive_pivot_candidate:
            status = 'Unbounded'
            logger.warning(f"{phase_name}: Không giới nội (biến vào {str(entering_var_sym)}).")
            step_counter += 1
            title_unbounded = f'{phase_name} - Bước {step_counter} (Vào: {str(entering_var_sym)}, Không giới nội)'
            steps_history[title_unbounded] = _get_ordered_tableau_for_history(current_tableau, objective_var_name)
            _print_tableau(title_unbounded, current_tableau, basic_vars, non_basic_vars, objective_var_name)
            break

        if not potential_leaving_vars:
            status = 'Error'
            logger.error(f"{phase_name}: Không tìm thấy biến ra cho {str(entering_var_sym)} (không có tỷ lệ hợp lệ).")
            break

        min_ratio_val = min(r for r, v_sym in potential_leaving_vars)
        tied_leaving_vars = [v_sym for r, v_sym in potential_leaving_vars if abs(r - min_ratio_val) < SIMPLEX_TOLERANCE]
        leaving_var_sym: Symbol
        if len(tied_leaving_vars) > 1 or is_degenerate:
            tied_leaving_vars.sort(key=get_bland_key)
            leaving_var_sym = tied_leaving_vars[0]
            logger.info(f"{phase_name} (Hòa/Suy biến): Chọn biến ra {str(leaving_var_sym)} (tỷ lệ: {min_ratio_val:.4f}).")
        else:
            leaving_var_sym = tied_leaving_vars[0]
            logger.info(f"{phase_name}: Chọn biến ra {str(leaving_var_sym)} (tỷ lệ: {min_ratio_val:.4f}).")

        step_counter += 1
        logger.info(f"{phase_name} - --- Phép xoay Bước {step_counter} --- Vào: {str(entering_var_sym)}, Ra: {str(leaving_var_sym)}")
        
        # Kiểm tra leaving_var_sym có trong current_tableau không
        if str(leaving_var_sym) not in current_tableau:
            logger.error(f"{phase_name}: Biến rời '{str(leaving_var_sym)}' không tìm thấy trong bảng. Dừng xoay.")
            status = 'Error'
            break
            
        pivot_row_expr_old = current_tableau[str(leaving_var_sym)]
        coeff_entering_in_pivot_row_expr = pivot_row_expr_old.coeff(entering_var_sym)
        if abs(float(coeff_entering_in_pivot_row_expr.evalf(chop=True))) < SIMPLEX_TOLERANCE / 100:
            status = 'Error'; logger.error(f"{phase_name}: LỖI Pivot gần 0."); break

        P_rest = simplify(pivot_row_expr_old - coeff_entering_in_pivot_row_expr * entering_var_sym)
        substitution_expr_for_entering_var = simplify((leaving_var_sym - P_rest) / coeff_entering_in_pivot_row_expr)
        
        new_tableau_temp = {} # Tạo bảng tạm thời để xây dựng
        # Dòng pivot mới (biến vào trở thành cơ bản)
        new_tableau_temp[str(entering_var_sym)] = substitution_expr_for_entering_var

        # Cập nhật các dòng khác, bao gồm cả dòng mục tiêu
        for var_name_iter, old_expr_iter in current_tableau.items():
            if var_name_iter != str(leaving_var_sym): # Không cần xử lý dòng biến rời cũ nữa
                 # Nếu var_name_iter là biến vào thì đã được xử lý ở trên, không cần thay thế chính nó
                if var_name_iter == str(entering_var_sym):
                    # Đối với dòng của biến vào (nay là dòng mục tiêu nếu tên trùng, hoặc dòng ràng buộc khác)
                    # nếu nó không phải là dòng pivot cũ, nó vẫn cần được cập nhật
                    # Tuy nhiên, logic này thường áp dụng cho các dòng *khác* dòng pivot và dòng biến vào mới
                    # Xem xét lại logic này: dòng entering_var_sym đã được gán biểu thức mới
                    # Các dòng khác (bao gồm cả objective_var_name nếu nó khác entering_var_sym) sẽ được thay thế
                    pass # Đã được xử lý bởi new_tableau_temp[str(entering_var_sym)] = ...
                else: # Các dòng khác (bao gồm cả dòng mục tiêu)
                    new_tableau_temp[var_name_iter] = simplify(old_expr_iter.subs(entering_var_sym, substitution_expr_for_entering_var))
        
        current_tableau = new_tableau_temp # Gán lại bảng đã cập nhật hoàn chỉnh

        basic_vars.remove(str(leaving_var_sym)); basic_vars.append(str(entering_var_sym))
        non_basic_vars.remove(str(entering_var_sym)); non_basic_vars.append(str(leaving_var_sym))
        basic_vars.sort(key=get_bland_key); non_basic_vars.sort(key=get_bland_key)
        basic_var_symbols = [Symbol(s) for s in basic_vars]; non_basic_var_symbols = [Symbol(s) for s in non_basic_vars]

        title_step_n = f'{phase_name} - Bước {step_counter} (Vào: {str(entering_var_sym)}, Ra: {str(leaving_var_sym)})'
        steps_history[title_step_n] = _get_ordered_tableau_for_history(current_tableau, objective_var_name)
        _print_tableau(title_step_n, current_tableau, basic_vars, non_basic_vars, objective_var_name)

    if iteration_count >= max_iterations: status = "MaxIterations"; logger.warning(f"{phase_name}: Đạt tối đa vòng lặp.")

    final_objective_value: Optional[float] = None
    final_solution_expressions: Dict[str, Expr] = {}
    if status == 'Optimal' or status == 'Multiple Optima':
        obj_expr_at_opt = current_tableau.get(objective_var_name, S.Zero)
        subs_final_nb_to_zero = {Symbol(nb_s): S.Zero for nb_s in non_basic_vars}
        try: final_objective_value = float(obj_expr_at_opt.subs(subs_final_nb_to_zero).evalf(chop=True))
        except Exception as e: logger.error(f"{phase_name}: Lỗi tính giá trị mục tiêu: {e}"); status = 'Error'

        for b_var_str in basic_vars:
            if b_var_str in current_tableau: # Đảm bảo biến cơ bản có trong bảng cuối cùng
                 final_solution_expressions[b_var_str] = current_tableau[b_var_str]
            else: # Xử lý trường hợp biến cơ bản không có trong bảng cuối (có thể do lỗi trước đó)
                 final_solution_expressions[b_var_str] = S.Zero # Hoặc một giá trị mặc định phù hợp
                 logger.warning(f"{phase_name}: Biến cơ bản {b_var_str} không tìm thấy trong bảng cuối, gán giá trị 0.")

        for nb_var_str in non_basic_vars:
            is_param = False
            if status == 'Multiple Optima' and objective_var_name in current_tableau: # Kiểm tra objective_var_name
                obj_coeff = current_tableau[objective_var_name].coeff(Symbol(nb_var_str))
                if obj_coeff.is_Number and abs(float(obj_coeff.evalf())) < SIMPLEX_TOLERANCE:
                    if any(
                        bvs in current_tableau and # Kiểm tra bvs
                        current_tableau[bvs].coeff(Symbol(nb_var_str)).is_Number and \
                        float(-current_tableau[bvs].coeff(Symbol(nb_var_str)).evalf()) > SIMPLEX_TOLERANCE
                        for bvs in basic_vars
                    ):
                        is_param = True
            final_solution_expressions[nb_var_str] = Symbol(nb_var_str) if is_param else S.Zero

    elif status == 'Unbounded': final_objective_value = float('-inf')
    return status, final_objective_value, final_solution_expressions, current_tableau, basic_vars, non_basic_vars, steps_history

# --- Hàm Simplex Hai Pha Chính ---
def simplex_two_phase(
    A_orig: List[List[float]], b_orig: List[float], c_orig: List[float],
    constraint_types_orig: List[str], variable_types_orig: List[str],
    objective_type_orig: str = 'max'
) -> Dict[str, Any]:
    num_original_vars = len(c_orig)
    num_constraints = len(b_orig)

    original_var_info_map: List[Dict[str, Any]] = []
    A_eff_cols: List[List[Expr]] = [[] for _ in range(num_constraints)]
    c_eff_sympy_list: List[Expr] = []
    decision_vars_transformed_symbols: List[Symbol] = []

    current_y_idx = 0
    for i in range(num_original_vars):
        original_name = f"x{i+1}"
        var_type = variable_types_orig[i]
        var_info = {'original_name': original_name, 'original_idx': i, 'type': var_type}
        if var_type == '<=0':
            y_name = f"y{current_y_idx + 1}"; y_sym = Symbol(y_name); current_y_idx += 1
            for r in range(num_constraints): A_eff_cols[r].append(-S(A_orig[r][i]))
            c_eff_sympy_list.append(-S(c_orig[i])); decision_vars_transformed_symbols.append(y_sym)
            var_info.update({'transformed_name': y_name, 'transformed_symbol': y_sym, 'is_negated': True})
        elif var_type == 'URS':
            p_name = f"{original_name}_p"; p_sym = Symbol(p_name)
            n_name = f"{original_name}_n"; n_sym = Symbol(n_name)
            for r in range(num_constraints): A_eff_cols[r].extend([S(A_orig[r][i]), -S(A_orig[r][i])])
            c_eff_sympy_list.extend([S(c_orig[i]), -S(c_orig[i])]); decision_vars_transformed_symbols.extend([p_sym, n_sym])
            var_info.update({'p_name': p_name, 'p_symbol': p_sym, 'n_name': n_name, 'n_symbol': n_sym, 'is_urs': True})
        else: # '>=0'
            x_sym = Symbol(original_name)
            for r in range(num_constraints): A_eff_cols[r].append(S(A_orig[r][i]))
            c_eff_sympy_list.append(S(c_orig[i])); decision_vars_transformed_symbols.append(x_sym)
            var_info.update({'transformed_name': original_name, 'transformed_symbol': x_sym, 'is_standard': True})
        original_var_info_map.append(var_info)

    A_eff = [[A_eff_cols[r][c] for c in range(len(decision_vars_transformed_symbols))] for r in range(num_constraints)]
    b_eff_exprs = [S(val) for val in b_orig]

    logger.info(f"Biến gốc đã xử lý: {original_var_info_map}")
    logger.info(f"Các biến quyết định đã biến đổi (thứ tự cột A_eff): {[str(s) for s in decision_vars_transformed_symbols]}")

    combined_steps: Dict[str, List[Tuple[str, str]]] = {} # Thay đổi kiểu của combined_steps

    logger.info("\n" + "="*10 + " Bắt đầu Pha 1 (Phương pháp x0) " + "="*10)
    x0_sym = Symbol('x0')
    decision_vars_for_phase1_tableau_symbols = decision_vars_transformed_symbols + [x0_sym]

    tableau_p1: Dict[str, Expr] = {}
    basic_vars_p1_initial: List[str] = []
    non_basic_vars_p1_initial: List[str] = [str(s) for s in decision_vars_transformed_symbols] + [str(x0_sym)]
    w_symbols: List[Symbol] = []

    for i in range(num_constraints):
        w_name = f"w{i+1}"; w_sym = Symbol(w_name); w_symbols.append(w_sym)
        basic_vars_p1_initial.append(w_name)

        expr_for_wj = x0_sym

        current_b_orig = b_eff_exprs[i]
        current_A_orig_coeffs = [A_eff[i][k_loop] for k_loop in range(len(decision_vars_transformed_symbols))]
        constraint_type = constraint_types_orig[i]

        if constraint_type == '>=':
            expr_for_wj += -current_b_orig
            for k_idx_wj, x_k_sym_wj_loop in enumerate(decision_vars_transformed_symbols):
                expr_for_wj += current_A_orig_coeffs[k_idx_wj] * x_k_sym_wj_loop
        elif constraint_type == '<=' or constraint_type == '=':
            expr_for_wj += current_b_orig
            for k_idx_wj, x_k_sym_wj_loop in enumerate(decision_vars_transformed_symbols):
                expr_for_wj -= current_A_orig_coeffs[k_idx_wj] * x_k_sym_wj_loop
        else:
            logger.error(f"Loại ràng buộc không xác định: {constraint_type} cho ràng buộc {i}")
            return {'status': 'Lỗi', 'z': "N/A", 'solution': {}, 'steps': {}, 'error_message': f"Loại ràng buộc không hợp lệ: {constraint_type}"}

        tableau_p1[w_name] = simplify(expr_for_wj)

    S_sym = Symbol('S'); tableau_p1[str(S_sym)] = x0_sym

    title_initial_wj_tableau = "Pha 1 - Bảng w_j ban đầu (trước tiền xử lý)"
    _print_tableau(title_initial_wj_tableau, tableau_p1, basic_vars_p1_initial, non_basic_vars_p1_initial, str(S_sym))
    # Lưu trữ bảng w_j ban đầu cho combined_steps với thứ tự dòng S lên đầu
    combined_steps[title_initial_wj_tableau] = _get_ordered_tableau_for_history(tableau_p1, str(S_sym))


    min_rhs_val, leaving_var_w_name = S.Zero, None
    for w_name_check in basic_vars_p1_initial:
        const_in_wj = tableau_p1[w_name_check].subs({s: S.Zero for s in decision_vars_for_phase1_tableau_symbols})
        if const_in_wj.is_Number and float(const_in_wj.evalf()) < float(min_rhs_val.evalf() - SIMPLEX_TOLERANCE):
            min_rhs_val, leaving_var_w_name = const_in_wj, w_name_check

    tableau_p1_pre_pivoted = {}
    for k,v in tableau_p1.items():
        if isinstance(v, Number): tableau_p1_pre_pivoted[k] = v
        elif hasattr(v, 'copy'):
            try: tableau_p1_pre_pivoted[k] = v.copy()
            except TypeError: tableau_p1_pre_pivoted[k] = sympify(v)
        else: tableau_p1_pre_pivoted[k] = v

    basic_vars_p1_pre_pivoted, non_basic_vars_p1_pre_pivoted = basic_vars_p1_initial[:], non_basic_vars_p1_initial[:]
    phase_name_for_core_solver = "Phase 1"

    if leaving_var_w_name is not None and float(min_rhs_val.evalf()) < -SIMPLEX_TOLERANCE:
        logger.info(f"Pha 1 (Tiền xử lý): x0 vào cơ sở, {leaving_var_w_name} rời cơ sở.")
        C_expr_for_prepivot = tableau_p1[leaving_var_w_name].subs({x0_sym: S.Zero})
        expr_for_x0_pivot_val = Symbol(leaving_var_w_name) - C_expr_for_prepivot
        expr_for_x0_pivot_val = simplify(expr_for_x0_pivot_val)

        temp_tableau_after_pre_pivot = {}
        temp_tableau_after_pre_pivot[str(x0_sym)] = expr_for_x0_pivot_val
        temp_tableau_after_pre_pivot[str(S_sym)] = expr_for_x0_pivot_val

        for w_iter_name in basic_vars_p1_initial:
            if w_iter_name != leaving_var_w_name:
                 temp_tableau_after_pre_pivot[w_iter_name] = simplify(tableau_p1[w_iter_name].subs(x0_sym, expr_for_x0_pivot_val))

        tableau_p1_pre_pivoted = temp_tableau_after_pre_pivot

        basic_vars_p1_pre_pivoted.remove(leaving_var_w_name); basic_vars_p1_pre_pivoted.append(str(x0_sym))
        non_basic_vars_p1_pre_pivoted.remove(str(x0_sym)); non_basic_vars_p1_pre_pivoted.append(leaving_var_w_name)
        basic_vars_p1_pre_pivoted.sort(key=get_bland_key); non_basic_vars_p1_pre_pivoted.sort(key=get_bland_key)
        phase_name_for_core_solver = "Phase 1 (Sau tiền xử lý x0)"
    else:
        logger.info("Pha 1: Không cần tiền xử lý pivot cho x0.")

    status_p1, min_S_value, sol_exprs_p1_core, final_tableau_p1, \
    final_basic_vars_p1, final_non_basic_vars_p1, steps_p1_from_core = _simplex_core_solver(
        tableau_p1_pre_pivoted, basic_vars_p1_pre_pivoted, non_basic_vars_p1_pre_pivoted,
        str(S_sym), phase_name_for_core_solver
    )

    if steps_p1_from_core: combined_steps.update(steps_p1_from_core)


    if status_p1 not in ['Optimal', 'Multiple Optima'] or \
       (min_S_value is not None and abs(min_S_value) > SIMPLEX_TOLERANCE and min_S_value > 0):
        error_msg_val = f"{min_S_value:.2f}" if min_S_value is not None else "không xác định"
        logger.error(f"Pha 1 (x0): Vô nghiệm (min S = {error_msg_val} > 0).")
        # combined_steps đã được cập nhật từ steps_p1_from_core
        return {'status': 'Vô nghiệm (Infeasible)', 'z': "N/A", 'solution': {}, 'steps': combined_steps,
                'error_message': f"Pha 1 (x0) kết thúc với min S = {error_msg_val} > 0."}

    if str(x0_sym) in final_basic_vars_p1 and str(x0_sym) in final_tableau_p1: # Thêm kiểm tra x0_sym trong final_tableau_p1
        x0_val_expr_at_end_p1 = final_tableau_p1[str(x0_sym)].subs({Symbol(nb): S.Zero for nb in final_non_basic_vars_p1})
        if x0_val_expr_at_end_p1.is_Number and abs(float(x0_val_expr_at_end_p1.evalf())) > SIMPLEX_TOLERANCE:
            logger.error(f"Pha 1 (x0): Lỗi! x0 cơ bản với giá trị {format_expression_for_printing(x0_val_expr_at_end_p1)} != 0.")
            # combined_steps đã được cập nhật
            return {'status': 'Vô nghiệm (Lỗi Pha 1)', 'z': "N/A", 'solution': {}, 'steps': combined_steps,
                    'error_message': f"Pha 1 (x0) lỗi: x0 cơ bản với giá trị {format_expression_for_printing(x0_val_expr_at_end_p1)}."}
        else:
            logger.info(f"Pha 1 (x0): x0 cơ bản với giá trị 0. Sẽ loại bỏ.")

    logger.info(f"Pha 1 (x0): Kết thúc (min S = {min_S_value if min_S_value is not None else 'N/A'}). Bắt đầu Pha 2.")
    logger.info("\n" + "="*10 + " Bắt đầu Pha 2 " + "="*10)
    tableau_p2, basic_vars_p2, non_basic_vars_p2 = {}, [], []

    for bv_p1_final in final_basic_vars_p1:
        if bv_p1_final != str(x0_sym) and not bv_p1_final.startswith('w'):
            basic_vars_p2.append(bv_p1_final)
    for nbv_p1_final in final_non_basic_vars_p1:
        if nbv_p1_final != str(x0_sym) and not nbv_p1_final.startswith('w'):
            non_basic_vars_p2.append(nbv_p1_final)

    all_transformed_dec_var_names_set = {str(s) for s in decision_vars_transformed_symbols}
    current_p2_vars_set = set(basic_vars_p2 + non_basic_vars_p2)
    for dec_var_name_check in all_transformed_dec_var_names_set:
        if dec_var_name_check not in current_p2_vars_set:
            logger.info(f"Biến quyết định {dec_var_name_check} không có trong cơ sở/phi cơ sở P2, thêm vào phi cơ sở P2 (giá trị 0).")
            if dec_var_name_check not in non_basic_vars_p2: non_basic_vars_p2.append(dec_var_name_check)

    basic_vars_p2 = sorted(list(set(basic_vars_p2)), key=get_bland_key)
    non_basic_vars_p2 = sorted(list(set(non_basic_vars_p2)), key=get_bland_key)

    subs_for_p2_tableau_build = {x0_sym: S.Zero}
    for w_nb_p1_build in final_non_basic_vars_p1:
        if w_nb_p1_build.startswith('w'): subs_for_p2_tableau_build[Symbol(w_nb_p1_build)] = S.Zero
    for w_b_p1_build in final_basic_vars_p1:
        if w_b_p1_build.startswith('w'): subs_for_p2_tableau_build[Symbol(w_b_p1_build)] = S.Zero

    for var_name_fp1, expr_fp1 in final_tableau_p1.items():
        if var_name_fp1 in basic_vars_p2 and var_name_fp1 != str(S_sym): # Không copy dòng S từ pha 1 vào ràng buộc pha 2
            tableau_p2[var_name_fp1] = simplify(expr_fp1.subs(subs_for_p2_tableau_build))


    is_max_problem = objective_type_orig.lower() == 'max'
    z_obj_expr_terms = S.Zero
    for i_coeff_z, coeff_val_z in enumerate(c_eff_sympy_list):
        z_obj_expr_terms += coeff_val_z * decision_vars_transformed_symbols[i_coeff_z]

    objective_name_p2 = 'z_obj'
    z_obj_expr_terms_no_x0 = z_obj_expr_terms.subs({x0_sym: S.Zero})
    # Thay thế các biến w (nếu có) bằng 0 trong biểu thức z_obj
    z_obj_expr_terms_no_w = z_obj_expr_terms_no_x0.subs(subs_for_p2_tableau_build)
    tableau_p2[objective_name_p2] = simplify(-z_obj_expr_terms_no_w if is_max_problem else z_obj_expr_terms_no_w)


    subs_basics_into_z_p2 = {Symbol(b_str_z): tableau_p2[b_str_z]
                           for b_str_z in basic_vars_p2
                           if Symbol(b_str_z) in tableau_p2[objective_name_p2].free_symbols and b_str_z in tableau_p2}
    if subs_basics_into_z_p2:
        tableau_p2[objective_name_p2] = simplify(tableau_p2[objective_name_p2].subs(subs_basics_into_z_p2))

    status_p2, z_solver_value, sol_exprs_p2, final_tableau_p2, \
    final_basic_vars_p2, final_non_basic_vars_p2, steps_p2_from_core = _simplex_core_solver(
        tableau_p2, basic_vars_p2, non_basic_vars_p2, objective_name_p2, "Phase 2")

    if steps_p2_from_core: combined_steps.update(steps_p2_from_core)


    final_z_value: Union[float, str, None] = "N/A"
    if z_solver_value is not None:
        if status_p2 == 'Unbounded': final_z_value = float('inf') if is_max_problem else float('-inf')
        elif status_p2 in ['Optimal', 'Multiple Optima']:
            raw_z = -z_solver_value if is_max_problem else z_solver_value
            final_z_value = 0.0 if abs(raw_z) < SIMPLEX_TOLERANCE else raw_z

    solution_final_orig_vars: Dict[str, Any] = {}
    if status_p2 in ['Optimal', 'Multiple Optima'] and sol_exprs_p2:
        final_subs_dict_p2_sol = {Symbol(nb_name_sol): expr_nb_sol
                               for nb_name_sol, expr_nb_sol in sol_exprs_p2.items()
                               if nb_name_sol in final_non_basic_vars_p2 and isinstance(expr_nb_sol, Expr)} # Thêm kiểm tra kiểu

        for var_map_info in original_var_info_map:
            orig_var_name = var_map_info['original_name']
            val_expr_for_orig_var: Expr = S.Zero
            if var_map_info['type'] == '>=0':
                transformed_sym = var_map_info['transformed_symbol']
                # Lấy biểu thức từ sol_exprs_p2, nếu không có thì từ final_tableau_p2 (nếu là biến cơ bản)
                expr_val = sol_exprs_p2.get(str(transformed_sym))
                if expr_val is None and str(transformed_sym) in final_tableau_p2: # Nếu là biến cơ bản
                     expr_val = final_tableau_p2[str(transformed_sym)]
                elif expr_val is None: # Nếu không phải cơ bản và không có trong sol_exprs_p2 (ví dụ, không phải tham số)
                     expr_val = S.Zero

                if isinstance(expr_val, Expr): # Chỉ thay thế nếu là biểu thức
                    val_expr_for_orig_var = expr_val.subs(final_subs_dict_p2_sol)
                else: # Nếu là S.Zero hoặc Symbol() từ sol_exprs_p2
                    val_expr_for_orig_var = expr_val


            elif var_map_info['type'] == '<=0':
                transformed_sym = var_map_info['transformed_symbol']
                expr_y_val = sol_exprs_p2.get(str(transformed_sym))
                if expr_y_val is None and str(transformed_sym) in final_tableau_p2:
                     expr_y_val = final_tableau_p2[str(transformed_sym)]
                elif expr_y_val is None:
                     expr_y_val = S.Zero

                if isinstance(expr_y_val, Expr):
                    val_expr_for_orig_var = -expr_y_val.subs(final_subs_dict_p2_sol)
                else:
                    val_expr_for_orig_var = -expr_y_val


            elif var_map_info['type'] == 'URS':
                p_sym, n_sym = var_map_info['p_symbol'], var_map_info['n_symbol']
                expr_p_val = sol_exprs_p2.get(str(p_sym))
                if expr_p_val is None and str(p_sym) in final_tableau_p2: expr_p_val = final_tableau_p2[str(p_sym)]
                elif expr_p_val is None: expr_p_val = S.Zero

                expr_n_val = sol_exprs_p2.get(str(n_sym))
                if expr_n_val is None and str(n_sym) in final_tableau_p2: expr_n_val = final_tableau_p2[str(n_sym)]
                elif expr_n_val is None: expr_n_val = S.Zero

                p_val_sub = expr_p_val.subs(final_subs_dict_p2_sol) if isinstance(expr_p_val, Expr) else expr_p_val
                n_val_sub = expr_n_val.subs(final_subs_dict_p2_sol) if isinstance(expr_n_val, Expr) else expr_n_val
                val_expr_for_orig_var = p_val_sub - n_val_sub

            solution_final_orig_vars[orig_var_name] = format_expression_for_printing(simplify(val_expr_for_orig_var))

    status_map_vn = {'Optimal': 'Tối ưu (Optimal)', 'Multiple Optima': 'Vô số nghiệm (Multiple Optima)',
                     'Unbounded': 'Không giới nội (Unbounded)', 'Infeasible': 'Vô nghiệm (Infeasible)',
                     'Error': 'Lỗi (Error)', 'MaxIterations': 'Đạt giới hạn vòng lặp (Max Iterations)'}
    final_status_str_vn = status_map_vn.get(status_p2 if status_p1 in ['Optimal', 'Multiple Optima'] or min_S_value is None or abs(min_S_value) < SIMPLEX_TOLERANCE else status_p1 , status_p2) # Ưu tiên status_p1 nếu nó là lỗi/vô nghiệm

    # Nếu Pha 1 không tối ưu (ví dụ: Vô nghiệm), thì kết quả cuối cùng là của Pha 1
    if not (status_p1 in ['Optimal', 'Multiple Optima'] and (min_S_value is None or abs(min_S_value) < SIMPLEX_TOLERANCE)):
        final_status_str_vn = status_map_vn.get(status_p1, status_p1)
        # Nếu Pha 1 vô nghiệm, giá trị z không xác định
        if status_p1 == 'Infeasible': final_z_value = "N/A"


    z_display_value: str
    if isinstance(final_z_value, str): z_display_value = final_z_value
    elif final_z_value is None: z_display_value = "N/A"
    else: z_display_value = "inf" if final_z_value == float('inf') else ("-inf" if final_z_value == float('-inf') else f"{final_z_value:.2f}")

    return {'status': final_status_str_vn, 'z': z_display_value, 'solution': solution_final_orig_vars,
            'steps': combined_steps, 'error_message': None}
