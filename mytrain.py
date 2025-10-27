"""
Enhanced Loan Default Prediction Training Script - FIXED VERSION

This script trains multiple models (Logistic Regression, Random Forest, SVM, CatBoost, LightGBM) 
with comprehensive feature engineering and Optuna hyperparameter optimization.

Usage:
    python loan_default_train.py --data-path data.csv --target target_default
    python loan_default_train.py --data-path data.csv --target target_default --model-type lightgbm --n-trials 50
    python loan_default_train.py --data-path data.csv --target target_default --model-type all --n-trials 100
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report)
import optuna
import mlflow
import mlflow.sklearn
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    import catboost
    from catboost import CatBoostClassifier
except ImportError:
    print("Installing CatBoost...")
    os.system("pip install catboost")
    import catboost
    from catboost import CatBoostClassifier

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    print("Installing LightGBM...")
    os.system("pip install lightgbm")
    import lightgbm as lgb
    from lightgbm import LGBMClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced loan default prediction with Optuna optimization")
    parser.add_argument("--data-path", type=str, default="loan_default_sample.csv", 
                       help="Path to CSV file")
    parser.add_argument("--target", type=str, default="target_default", 
                       help="Target column name")
    parser.add_argument("--model-type", type=str, 
                       choices=["logistic", "random_forest", "svm", "catboost", "lightgbm", "all"], 
                       default="all", help="Model type to train")
    parser.add_argument("--n-trials", type=int, default=50, 
                       help="Number of Optuna optimization trials")
    parser.add_argument("--cv-folds", type=int, default=5, 
                       help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, 
                       help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="Loan-Default-Optuna-Optimization")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=None, 
                       help="Optimization timeout in seconds")
    return parser.parse_args()


def add_comprehensive_features(df):
    """
    Comprehensive feature engineering for loan default prediction
    """
    print("🔧 Performing feature engineering...")
    
    # Basic ratios and risk indicators
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1e-8)
    df['monthly_payment'] = df['loan_amount'] / (df['term_months'] + 1e-8)
    df['payment_to_income_ratio'] = (df['monthly_payment'] * 12) / (df['annual_income'] + 1e-8)
    
    # Employment and age risk factors
    df['employment_risk'] = (df['employment_length'] < 2).astype(int)
    df['young_borrower'] = (df['age'] < 30).astype(int)
    df['senior_borrower'] = (df['age'] > 60).astype(int)
    df['experienced_worker'] = (df['employment_length'] > 10).astype(int)
    
    # Credit and financial health indicators
    df['high_credit_score'] = (df['credit_score'] > 750).astype(int)
    df['low_credit_score'] = (df['credit_score'] < 580).astype(int)
    df['high_interest'] = (df['interest_rate'] > df['interest_rate'].median()).astype(int)
    df['multiple_delinquencies'] = (df['delinquency_2yrs'] > 1).astype(int)
    df['many_open_accounts'] = (df['num_open_acc'] > df['num_open_acc'].median()).astype(int)
    
    # Credit utilization and debt management
    if 'total_acc' in df.columns and 'num_open_acc' in df.columns:
        df['account_utilization'] = df['num_open_acc'] / (df['total_acc'] + 1e-8)
    
    # Credit score binning
    df['credit_score_binned'] = pd.cut(
        df['credit_score'], 
        bins=[0, 580, 670, 740, 850], 
        labels=['Poor', 'Fair', 'Good', 'Excellent'],
        include_lowest=True
    )
    
    # Income and loan amount binning
    df['income_binned'] = pd.qcut(df['annual_income'], q=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'], duplicates='drop')
    df['loan_amount_binned'] = pd.qcut(df['loan_amount'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'], duplicates='drop')
    
    # Risk scoring system
    risk_factors = ['employment_risk', 'young_borrower', 'high_interest', 
                   'multiple_delinquencies', 'many_open_accounts', 'low_credit_score']
    df['risk_score'] = df[risk_factors].sum(axis=1)
    
    # Positive factors
    positive_factors = ['high_credit_score', 'experienced_worker']
    df['positive_score'] = df[positive_factors].sum(axis=1)
    
    # Combined risk assessment
    df['net_risk_score'] = df['risk_score'] - df['positive_score']
    
    # Interaction features
    df['income_age_interaction'] = df['annual_income'] * df['age']
    df['credit_employment_interaction'] = df['credit_score'] * df['employment_length']
    
    print(f"✅ Feature engineering completed. Total features: {df.shape[1]}")
    return df


def create_optuna_objectives(preprocessor):
    """Create Optuna objective functions for different models with preprocessor"""
    
    def logistic_objective(trial, X_train, y_train, cv):
        # Hyperparameter suggestions
        C = trial.suggest_float('C', 1e-4, 100, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        # Create pipeline with preprocessor
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=C, penalty=penalty, solver=solver, class_weight=class_weight,
                random_state=42, max_iter=1000
            ))
        ])
        
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    def rf_objective(trial, X_train, y_train, cv):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                max_features=max_features, class_weight=class_weight,
                random_state=42, n_jobs=-1
            ))
        ])
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    def svm_objective(trial, X_train, y_train, cv):
        C = trial.suggest_float('C', 1e-2, 100, log=True)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                C=C, kernel=kernel, gamma=gamma, class_weight=class_weight,
                random_state=42, probability=True
            ))
        ])
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    def catboost_objective(trial, X_train, y_train, cv):
        depth = trial.suggest_int('depth', 4, 10)
        iterations = trial.suggest_int('iterations', 100, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', CatBoostClassifier(
                depth=depth, iterations=iterations, learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg, random_state=42, verbose=False, 
                auto_class_weights='Balanced'
            ))
        ])
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    def lightgbm_objective(trial, X_train, y_train, cv):
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        num_leaves = trial.suggest_int('num_leaves', 10, 300)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 100)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                num_leaves=num_leaves, min_child_samples=min_child_samples,
                subsample=subsample, colsample_bytree=colsample_bytree,
                random_state=42, verbose=-1, class_weight='balanced'
            ))
        ])
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        return scores.mean()
    
    return {
        'logistic': logistic_objective,
        'random_forest': rf_objective,
        'svm': svm_objective,
        'catboost': catboost_objective,
        'lightgbm': lightgbm_objective
    }


def optimize_model(model_name, objective_func, X_train, y_train, cv, n_trials, timeout):
    """Optimize a single model using Optuna"""
    print(f"\n🔍 Optimizing {model_name.upper()}...")
    
    # Create objective function with fixed parameters
    def objective(trial):
        return objective_func(trial, X_train, y_train, cv)
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    print(f"✅ {model_name.upper()} optimization completed")
    print(f"📊 Best ROC-AUC: {study.best_value:.4f}")
    print(f"🏆 Best parameters: {study.best_params}")
    
    return study.best_params, study.best_value


def create_final_model(model_name, best_params, preprocessor):
    """Create the final model pipeline with best parameters"""
    if model_name == 'logistic':
        classifier = LogisticRegression(**best_params, random_state=42, max_iter=1000)
    elif model_name == 'random_forest':
        classifier = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    elif model_name == 'svm':
        classifier = SVC(**best_params, random_state=42, probability=True)
    elif model_name == 'catboost':
        classifier = CatBoostClassifier(**best_params, random_state=42, verbose=False)
    elif model_name == 'lightgbm':
        classifier = LGBMClassifier(**best_params, random_state=42, verbose=-1)
    
    # Return pipeline with preprocessor
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n📊 {model_name.upper()} Final Performance:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def main():
    args = parse_args()
    
    # MLflow setup
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    
    print("🚀 Starting Enhanced Loan Default Prediction Training")
    print(f"📁 Data path: {args.data_path}")
    print(f"🎯 Target: {args.target}")
    print(f"🤖 Model(s): {args.model_type}")
    print(f"🔍 Optuna trials: {args.n_trials}")
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Feature engineering
    df = add_comprehensive_features(df)
    
    # Prepare target
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")
    
    # Clean data
    df = df.dropna(subset=[args.target])
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
    
    # Separate features and target
    X = df.drop(columns=[args.target])
    y = df[args.target]
    
    print(f"📊 Class distribution: {dict(y.value_counts())}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"📊 Numeric features: {len(numeric_features)}")
    print(f"📊 Categorical features: {len(categorical_features)}")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")
    
    # Determine models to train
    if args.model_type == "all":
        models_to_train = ['logistic', 'random_forest', 'svm', 'catboost', 'lightgbm']
    else:
        models_to_train = [args.model_type]
    
    # Get objective functions with preprocessor
    objectives = create_optuna_objectives(preprocessor)
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    
    # Store results
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Train each model
    for model_name in models_to_train:
        with mlflow.start_run(run_name=f"optuna_{model_name}_{timestamp}"):
            print(f"\n{'='*60}")
            print(f"🤖 TRAINING {model_name.upper()}")
            print(f"{'='*60}")
            
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("cv_folds", args.cv_folds)
            mlflow.log_param("test_size", args.test_size)
            
            # Optimize hyperparameters
            best_params, best_cv_score = optimize_model(
                model_name, objectives[model_name], 
                X_train, y_train, cv, args.n_trials, args.timeout
            )
            
            # Log best parameters and CV score
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_roc_auc", best_cv_score)
            
            # Create and train final model
            final_model = create_final_model(model_name, best_params, preprocessor)
            final_model.fit(X_train, y_train)
            
            # Evaluate on test set
            metrics = evaluate_model(final_model, X_test, y_test, model_name)
            
            # Log test metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Store results
            results[model_name] = {
                'model': final_model,
                'metrics': metrics,
                'best_params': best_params,
                'cv_score': best_cv_score
            }
            
            # Track best model
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = final_model
                best_model_name = model_name
            
            # Log model
            mlflow.sklearn.log_model(final_model, f"{model_name}_model")
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Results table
    print(f"{'Model':<15} {'CV ROC-AUC':<12} {'Test ROC-AUC':<13} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        cv_score = result['cv_score']
        test_metrics = result['metrics']
        print(f"{name:<15} {cv_score:<12.4f} {test_metrics['roc_auc']:<13.4f} "
              f"{test_metrics['accuracy']:<10.4f} {test_metrics['f1_score']:<10.4f}")
    
    print(f"\n🥇 BEST MODEL: {best_model_name.upper()}")
    print(f"🎯 Best Test ROC-AUC: {best_score:.4f}")
    
    # Save best model
    if best_model:
        export_dir = os.path.abspath("best_loan_model")
        if os.path.exists(export_dir):
            import shutil
            shutil.rmtree(export_dir)
        
        mlflow.sklearn.save_model(best_model, export_dir)
        print(f"💾 Best model saved to: {export_dir}")
        
        # Save comprehensive results
        summary = {
            'best_model': best_model_name,
            'best_test_roc_auc': best_score,
            'all_results': {
                name: {
                    'cv_score': result['cv_score'],
                    'test_metrics': result['metrics'],
                    'best_params': result['best_params']
                } for name, result in results.items()
            },
            'training_config': {
                'n_trials': args.n_trials,
                'cv_folds': args.cv_folds,
                'test_size': args.test_size,
                'random_state': args.random_state
            },
            'timestamp': timestamp
        }
        
        with open("optimization_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("📊 Complete results saved to: optimization_results.json")
    
    print("\n🎉 Training completed successfully!")
    print(f"📈 View MLflow results: mlflow ui")
    print(f"🚀 Load best model: mlflow.sklearn.load_model('best_loan_model')")


if __name__ == "__main__":
    main()