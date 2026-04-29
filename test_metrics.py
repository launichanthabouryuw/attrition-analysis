import pytest
import pandas as pd
from metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)
from load_data import clean_employee_data


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_df():
    """Four employees across two departments with mixed attrition."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
        "overtime": ["Yes", "No", "No", "Yes"],
        "job_satisfaction": [1, 3, 3, 2],
        "monthly_income": [4000, 6000, 5000, 7000],
        "travel_frequency": ["Frequent", "Rarely", "Rarely", "Occasional"],
        "years_at_company": [1, 5, 8, 3],
        "age": [25, 35, 40, 30],
    })


# ---------------------------------------------------------------------------
# attrition_rate
# ---------------------------------------------------------------------------

def test_attrition_rate_half(basic_df):
    assert attrition_rate(basic_df) == 50.0


def test_attrition_rate_zero_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["No", "No"],
    })
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["Yes", "Yes"],
    })
    assert attrition_rate(df) == 100.0


def test_attrition_rate_rounds_to_two_decimals():
    # 1 leaver out of 3 = 33.333...% -> should round to 33.33
    df = pd.DataFrame({
        "employee_id": [1, 2, 3],
        "attrition": ["Yes", "No", "No"],
    })
    assert attrition_rate(df) == 33.33


# ---------------------------------------------------------------------------
# attrition_by_department
# ---------------------------------------------------------------------------

def test_attrition_by_department_columns(basic_df):
    result = attrition_by_department(basic_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(basic_df):
    result = attrition_by_department(basic_df)
    rates = dict(zip(result["department"], result["attrition_rate"]))
    # Sales: 1 leaver / 2 employees = 50%
    # HR:    1 leaver / 2 employees = 50%
    assert rates["Sales"] == 50.0
    assert rates["HR"] == 50.0


def test_attrition_by_department_counts(basic_df):
    result = attrition_by_department(basic_df)
    counts = dict(zip(result["department"], result["employees"]))
    assert counts["Sales"] == 2
    assert counts["HR"] == 2


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "department": ["Sales", "Sales", "HR", "HR", "HR"],
        "attrition": ["Yes", "Yes", "Yes", "No", "No"],
    })
    result = attrition_by_department(df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# attrition_by_overtime
# ---------------------------------------------------------------------------

def test_attrition_by_overtime_columns(basic_df):
    result = attrition_by_overtime(basic_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(basic_df):
    result = attrition_by_overtime(basic_df)
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    # Overtime Yes: both employees left -> 100%
    # Overtime No:  neither left        -> 0%
    assert rates["Yes"] == 100.0
    assert rates["No"] == 0.0


def test_attrition_by_overtime_employee_counts(basic_df):
    result = attrition_by_overtime(basic_df)
    counts = dict(zip(result["overtime"], result["employees"]))
    assert counts["Yes"] == 2
    assert counts["No"] == 2


# ---------------------------------------------------------------------------
# average_income_by_attrition
# ---------------------------------------------------------------------------

def test_average_income_by_attrition_columns(basic_df):
    result = average_income_by_attrition(basic_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(basic_df):
    result = average_income_by_attrition(basic_df)
    income = dict(zip(result["attrition"], result["avg_monthly_income"]))
    # Leavers: employee 1 (4000) and 4 (7000) -> mean 5500
    # Stayers: employee 2 (6000) and 3 (5000) -> mean 5500
    assert income["Yes"] == 5500.0
    assert income["No"] == 5500.0


def test_average_income_leavers_earn_less():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "attrition": ["Yes", "Yes", "No", "No"],
        "monthly_income": [3000, 4000, 7000, 8000],
    })
    result = average_income_by_attrition(df)
    income = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert income["Yes"] < income["No"]


# ---------------------------------------------------------------------------
# satisfaction_summary
# ---------------------------------------------------------------------------

def test_satisfaction_summary_columns(basic_df):
    result = satisfaction_summary(basic_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rates_use_group_size_not_total_leavers():
    # satisfaction 1: 2 employees, 2 leavers -> 100%
    # satisfaction 3: 2 employees, 0 leavers -> 0%
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "attrition": ["Yes", "Yes", "No", "No"],
        "job_satisfaction": [1, 1, 3, 3],
    })
    result = satisfaction_summary(df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 100.0
    assert rates[3] == 0.0


def test_satisfaction_summary_sorted_ascending(basic_df):
    result = satisfaction_summary(basic_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)


def test_satisfaction_summary_partial_attrition():
    # satisfaction 2: 1 leaver out of 2 employees = 50%
    df = pd.DataFrame({
        "employee_id": [1, 2, 3],
        "attrition": ["Yes", "No", "No"],
        "job_satisfaction": [2, 2, 3],
    })
    result = satisfaction_summary(df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[2] == 50.0
    assert rates[3] == 0.0


# ---------------------------------------------------------------------------
# clean_employee_data
# ---------------------------------------------------------------------------

def _minimal_df(**overrides):
    """Build a one-row DataFrame with all required columns, optionally overriding values."""
    base = {
        "employee_id": [1],
        "department": ["Sales"],
        "age": [30],
        "monthly_income": [5000.0],
        "job_satisfaction": [3],
        "overtime": ["No"],
        "travel_frequency": ["Rarely"],
        "years_at_company": [5],
        "attrition": ["Yes"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_raises_on_missing_column():
    df = pd.DataFrame({"employee_id": [1], "department": ["Sales"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_fills_missing_department():
    df = _minimal_df(department=[None])
    result = clean_employee_data(df)
    assert result["department"].iloc[0] == "Unknown"


def test_clean_fills_missing_overtime():
    df = _minimal_df(overtime=[None])
    result = clean_employee_data(df)
    assert result["overtime"].iloc[0] == "No"


def test_clean_fills_missing_travel_frequency():
    df = _minimal_df(travel_frequency=[None])
    result = clean_employee_data(df)
    assert result["travel_frequency"].iloc[0] == "Rarely"


def test_clean_fills_missing_job_satisfaction():
    df = _minimal_df(job_satisfaction=[None])
    result = clean_employee_data(df)
    assert result["job_satisfaction"].iloc[0] == 3


def test_clean_fills_missing_monthly_income_with_median():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3],
        "department": ["Sales", "HR", "IT"],
        "age": [30, 35, 40],
        "monthly_income": [4000.0, None, 6000.0],
        "job_satisfaction": [3, 3, 3],
        "overtime": ["No", "No", "No"],
        "travel_frequency": ["Rarely", "Rarely", "Rarely"],
        "years_at_company": [5, 5, 5],
        "attrition": ["No", "No", "No"],
    })
    result = clean_employee_data(df)
    assert result["monthly_income"].iloc[1] == 5000.0


def test_clean_normalises_attrition_to_title_case():
    df = _minimal_df(attrition=["yes"])
    result = clean_employee_data(df)
    assert result["attrition"].iloc[0] == "Yes"


def test_clean_strips_whitespace_from_department():
    df = _minimal_df(department=["  Sales  "])
    result = clean_employee_data(df)
    assert result["department"].iloc[0] == "Sales"


def test_clean_does_not_mutate_original():
    df = _minimal_df(department=[None])
    original_val = df["department"].iloc[0]
    clean_employee_data(df)
    assert df["department"].iloc[0] is original_val
