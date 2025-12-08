def hybrid_predict(symptom_text, expert, logistic):
    """
    Chạy Expert System trước, sau đó Logistic Regression để refine.
    Trả về kết quả kết hợp từ cả hai mô hình.
    """
    # Expert System
    facts = {}
    new_syms = expert.extract_symptoms(symptom_text, facts)
    if isinstance(new_syms, set):
        for sym, val in new_syms:
            facts[sym] = val
    ranked = expert.forward_chaining(facts)
    top_expert = ranked[0] if ranked else None

    # Logistic Regression
    logistic_pred = logistic.predict(symptom_text)

    # Kết hợp
    if top_expert and logistic_pred:
        avg_score = (top_expert['percent'] + logistic_pred['percent']) / 2
        final_disease = (
            top_expert['name']
            if avg_score >= logistic_pred['percent']
            else logistic_pred['name']
        )
        return {
            "expert": top_expert,
            "logistic": logistic_pred,
            "final": {
                "name": final_disease,
                "score": avg_score
            },
        }
    else:
        return {
            "expert": top_expert,
            "logistic": logistic_pred,
            "final": None,
        }
