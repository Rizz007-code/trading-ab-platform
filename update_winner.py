from datetime import date
from data.database.connection import get_db_session
from data.database.models import Experiment, ExperimentResult

with get_db_session() as session:
    if session.query(ExperimentResult).count() == 0:
        exp = Experiment(
            name="Dummy_Exp",
            strategy_a="Strategy_A",
            strategy_b="Strategy_B",
            ticker="AAPL",
            start_date="2022-01-01",
            end_date="2024-01-01",
            status="completed"
        )
        session.add(exp)
        session.flush()
        
        exp_res = ExperimentResult(
            experiment_id=exp.id,
            winner="Strategy_B",
            lift_pct=10.0,
            p_value=0.01,
            is_significant=True,
            confidence_level=0.95,
            sharpe_a=1.0,
            annual_return_a=0.1,
            volatility_a=0.1,
            max_drawdown_a=0.1,
            win_rate_a=0.5,
            sharpe_b=1.5,
            annual_return_b=0.15,
            volatility_b=0.1,
            max_drawdown_b=0.05,
            win_rate_b=0.6,
            ci_lower=1.0,
            ci_upper=2.0,
            test_method="mock"
        )
        session.add(exp_res)
        session.commit()
        print("Mock experiment inserted.")
    else:
        print("Experiment already exists.")
