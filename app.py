from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Blueprint
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Transaction, Budget, Category
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename 
import camelot
import pandas as pd
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy import func
from dateutil.relativedelta import relativedelta
import statistics
import os
from parsers.bank_parser import BankStatementParser
from ml.predictor import CategoryPredictor
from transaction_processor import TransactionProcessor

# Initialize Flask app first
app = Flask(__name__)

# Then configure it
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://spenduser:password@localhost/spendsense'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure upload folder AFTER app is created
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.context_processor
def inject_datetime():
    return {"datetime": datetime} 

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    if current_user.is_authenticated:
        if not current_user.has_completed_setup:
            user_categories = Category.query.filter_by(user_id=current_user.id).first()
            if not user_categories:
                return redirect(url_for('select_categories'))
            return redirect(url_for('setup_budget'))
        
        # Go to dashboard properly
        return redirect(url_for('dashboard'))

    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email, has_completed_setup=False)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('select_categories'))
    
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Get selected month/year or use current
        selected_month = request.args.get('month', default=datetime.now().month, type=int)
        selected_year = request.args.get('year', default=datetime.now().year, type=int)
        
        # Validate month/year range
        if not (1 <= selected_month <= 12):
            selected_month = datetime.now().month
        if selected_year < 2000 or selected_year > datetime.now().year + 1:
            selected_year = datetime.now().year

        # 1. INCOME CALCULATION (only credits) - with error handling
        try:
            income = db.session.query(
                func.sum(Transaction.amount)
            ).filter(
                Transaction.user_id == current_user.id,
                Transaction.type == 'credit',
                db.extract('month', Transaction.date) == selected_month,
                db.extract('year', Transaction.date) == selected_year
            ).scalar() or 0
        except Exception as e:
            print(f"Income calculation error: {e}")
            income = 0

        # 2. EXPENSES CALCULATION (only debits) - with error handling
        try:
            expenses = db.session.query(
                func.sum(Transaction.amount)
            ).filter(
                Transaction.user_id == current_user.id,
                Transaction.type == 'debit',
                db.extract('month', Transaction.date) == selected_month,
                db.extract('year', Transaction.date) == selected_year
            ).scalar() or 0
        except Exception as e:
            print(f"Expenses calculation error: {e}")
            expenses = 0

        # 3. NET WORTH (all-time) - with error handling
        try:
            all_time_income = db.session.query(
                func.sum(Transaction.amount)
            ).filter(
                Transaction.user_id == current_user.id,
                Transaction.type == 'credit'
            ).scalar() or 0

            all_time_expenses = db.session.query(
                func.sum(Transaction.amount)
            ).filter(
                Transaction.user_id == current_user.id,
                Transaction.type == 'debit'
            ).scalar() or 0

            net_worth = all_time_income - all_time_expenses
        except Exception as e:
            print(f"Net worth calculation error: {e}")
            net_worth = 0

        # 4. SPENDING DATA - with better error handling
        try:
            expense_categories = Category.query.filter(
                ((Category.user_id == current_user.id) | (Category.is_default == True)),
                Category.is_income == False
            ).all()

            # If no categories exist, create a default one
            if not expense_categories:
                default_category = Category(
                    user_id=current_user.id,
                    name='Other',
                    color='#808080',
                    icon='tag',
                    is_income=False
                )
                db.session.add(default_category)
                db.session.commit()
                expense_categories = [default_category]

            budgets = {b.category_id: b for b in Budget.query.filter_by(user_id=current_user.id).all()}
            spending_data = {}
            category_totals = {}
            total_spent = 0

            for category in expense_categories:
                try:
                    category_spent = db.session.query(
                        func.sum(Transaction.amount)
                    ).filter(
                        Transaction.user_id == current_user.id,
                        Transaction.category_id == category.id,
                        Transaction.type == 'debit',
                        db.extract('month', Transaction.date) == selected_month,
                        db.extract('year', Transaction.date) == selected_year
                    ).scalar() or 0

                    budget_limit = budgets.get(category.id, Budget(limit=0)).limit
                    
                    spending_data[category.name] = {
                        'spent': category_spent,
                        'limit': budget_limit,
                        'remaining': max(0, budget_limit - category_spent),
                        'color': category.color,
                        'icon': category.icon
                    }
                    
                    total_spent += category_spent
                    category_totals[category.name] = category_spent
                    
                except Exception as e:
                    print(f"Error processing category {category.name}: {e}")
                    # Add default values for this category
                    spending_data[category.name] = {
                        'spent': 0,
                        'limit': 0,
                        'remaining': 0,
                        'color': category.color,
                        'icon': category.icon
                    }
                    category_totals[category.name] = 0

        except Exception as e:
            print(f"Spending data error: {e}")
            spending_data = {'Other': {'spent': 0, 'limit': 0, 'remaining': 0, 'color': '#808080', 'icon': 'tag'}}
            category_totals = {'Other': 0}
            total_spent = 0
            expense_categories = []

        # 5. BUDGET UTILIZATION - with error handling
        try:
            budget_categories = [b for b in budgets.values() if b.limit > 0]
            if budget_categories:
                budget_utilization = sum(
                    min(100, (spending_data.get(b.category.name, {}).get('spent', 0) / b.limit) * 100) 
                    for b in budget_categories
                ) / len(budget_categories)
            else:
                budget_utilization = 0
        except Exception as e:
            print(f"Budget utilization error: {e}")
            budget_utilization = 0

        # 6. TOP SPENDING CATEGORY - with error handling
        try:
            if category_totals and any(v > 0 for v in category_totals.values()):
                top_category = max(category_totals, key=category_totals.get)
                top_category_amount = category_totals[top_category]
            else:
                top_category = "No spending yet"
                top_category_amount = 0
        except Exception as e:
            print(f"Top category error: {e}")
            top_category = "Error calculating"
            top_category_amount = 0

        # 7. MONTHLY SPENDING TREND - with error handling
        try:
            monthly_spending = get_monthly_spending(current_user.id)
        except Exception as e:
            print(f"Monthly spending error: {e}")
            monthly_spending = {'labels': [], 'data': []}

        # 8. RECENT TRANSACTIONS - with error handling
        try:
            recent_transactions = Transaction.query.filter_by(
                user_id=current_user.id
            ).order_by(Transaction.date.desc()).limit(5).all()
        except Exception as e:
            print(f"Recent transactions error: {e}")
            recent_transactions = []

        # 9. SPENDING INSIGHTS - with error handling
        try:
            insights = generate_insights(current_user.id, selected_month, selected_year)
        except Exception as e:
            print(f"Insights error: {e}")
            insights = []

        # 10. SPENDING PROJECTIONS - with error handling
        try:
            projections = calculate_projections(current_user.id, selected_month, selected_year)
        except Exception as e:
            print(f"Projections error: {e}")
            projections = {}

        return render_template('dashboard.html',
            income=income,
            expenses=expenses,
            net_worth=net_worth,
            spending_data=spending_data,
            total_spent=total_spent,
            budget_utilization=round(budget_utilization, 1),
            top_category=top_category,
            top_category_amount=top_category_amount,
            insights=insights,
            recent_transactions=recent_transactions,
            monthly_spending=monthly_spending,
            projections=projections,
            categories=expense_categories,
            selected_month=selected_month,
            selected_year=selected_year,
            now=datetime.now(),
            datetime=datetime,
            timedelta=timedelta,
            abs=abs
        )

    except Exception as e:
        app.logger.error(f"Dashboard error: {str(e)}")
        print(f"Full dashboard error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Return a minimal dashboard if everything fails
        return render_template('dashboard.html',
            income=0,
            expenses=0,
            net_worth=0,
            spending_data={},
            total_spent=0,
            budget_utilization=0,
            top_category="No data",
            top_category_amount=0,
            insights=[],
            recent_transactions=[],
            monthly_spending={'labels': [], 'data': []},
            projections={},
            categories=[],
            selected_month=datetime.now().month,
            selected_year=datetime.now().year,
            now=datetime.now(),
            datetime=datetime,
            timedelta=timedelta,
            abs=abs
        )

#@app.route('/transactions')
#@login_required
#def transactions():
 #   transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).all()
  #  return render_template('transactions.html', transactions=transactions)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('transactions'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('transactions'))
    
    try:
        # Save temp file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Determine file type
        file_type = 'pdf' if filename.lower().endswith('.pdf') else 'csv'
        
        # Process file
        processor = TransactionProcessor()
        count = processor.process_uploaded_file(
            user_id=current_user.id,
            file_path=temp_path,
            file_type=file_type
        )
        
        flash(f'Successfully imported {count} transactions!', 'success')
        return redirect(url_for('transactions'))
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('transactions'))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Modify your add_transaction route
@app.route('/add-transaction', methods=['POST'])
@login_required
def add_transaction():
    description = request.form['description']
    amount = float(request.form['amount'])
    date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
    category_id = int(request.form['category_id'])
    
    # Verify category belongs to user
    category = Category.query.filter_by(
        id=category_id,
        user_id=current_user.id
    ).first_or_404()
    
    transaction = Transaction(
        user_id=current_user.id,
        category_id=category_id,
        date=date,
        description=description,
        amount=amount,
        type='debit' if amount > 0 else 'credit'
    )
    db.session.add(transaction)
    db.session.commit()
    
    flash('Transaction added successfully!')
    return redirect(url_for('transactions'))

# Update the upload processing
def process_uploaded_transactions(user_id, csv_path):
    processor = TransactionProcessor()
    transactions = processor.process_file(csv_path)
    
    for t in transactions:
        # Predict category using your ML model
        predicted_category = predict_category(t['description'], t['amount'])
        
        # Find or create category
        category = Category.query.filter_by(
            user_id=user_id,
            name=predicted_category
        ).first()
        
        if not category:
            # Create new category with default settings
            category = Category(
                user_id=user_id,
                name=predicted_category,
                is_default=False
            )
            db.session.add(category)
            db.session.flush()  # Get the ID before commit
        
        # Create transaction
        transaction = Transaction(
            user_id=user_id,
            category_id=category.id,
            date=t['date'],
            description=t['description'],
            amount=t['amount'],
            type=t['type']
        )
        db.session.add(transaction)
    
    db.session.commit()
    return len(transactions)

@app.route('/transactions')
@login_required
def transactions():
    # Get all categories (including defaults) for the current user
    categories = Category.query.filter(
        (Category.user_id == current_user.id) | 
        (Category.is_default == True)
    ).order_by(Category.name).all()

    # Get filter parameters from request
    category_id = request.args.get('category')
    month = request.args.get('month')
    search = request.args.get('search')

    # Base query - all transactions for current user
    query = Transaction.query.filter_by(user_id=current_user.id)

    # Apply category filter if specified
    if category_id and category_id.isdigit():
        query = query.filter_by(category_id=int(category_id))

    # Apply month filter if specified
    if month and month.isdigit():
        query = query.filter(db.extract('month', Transaction.date) == int(month))

    # Apply search filter if specified
    if search:
        query = query.filter(Transaction.description.ilike(f'%{search}%'))

    # Finalize query with sorting
    transactions = query.order_by(Transaction.date.desc()).all()

    return render_template(
        'transactions.html',
        transactions=transactions,
        categories=categories,
        datetime=datetime  # Pass datetime for template filters
    )

@app.route('/budget-setup', methods=['GET', 'POST'])
@login_required 
def budget_setup():
    if request.method == 'POST':
        budget_method = request.form.get('budget_method')
        
        if budget_method == 'auto':
            # Auto budget calculation
            three_months_ago = datetime.now() - relativedelta(months=3)
            Budget.query.filter_by(user_id=current_user.id).delete()
            
            category_spending = db.session.query(
                Category.id,
                func.avg(Transaction.amount).label('avg_spending')
            ).join(Transaction).filter(
                Transaction.user_id == current_user.id,
                Transaction.type == 'debit',
                Transaction.date >= three_months_ago
            ).group_by(Category.id).all()
            
            for category_id, avg_spending in category_spending:
                if avg_spending > 0:
                    budget = Budget(
                        user_id=current_user.id,
                        category_id=category_id,
                        limit=avg_spending * 1.2,
                        period='monthly'
                    )
                    db.session.add(budget)
            
            # Mark setup complete
            current_user.has_completed_setup = True
            db.session.commit()
            return redirect(url_for('dashboard'))
            
        elif budget_method == 'manual':
            return redirect(url_for('edit_budgets', clean_slate=True))
        else:
            flash('Invalid budget setup method', 'error')
            return redirect(url_for('budget_setup'))
    
    # GET request
    return render_template('budget_setup.html')

def auto_calculate_budgets(user_id):
    pass

def create_default_budgets(user_id):
    pass

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

def generate_insights(user_id, selected_month, selected_year):
    """Generate accurate spending insights"""
    insights = []
    
    # 1. Get all relevant transactions
    transactions = Transaction.query.filter(
        Transaction.user_id == user_id,
        Transaction.type == 'debit',  # Only expenses
        db.extract('month', Transaction.date) == selected_month,
        db.extract('year', Transaction.date) == selected_year
    ).all()
    
    # 2. Calculate total spending
    total_spent = sum(t.amount for t in transactions)
    
    if not transactions:
        return insights
    
    # 3. Top Spending Category
    by_category = defaultdict(float)
    for t in transactions:
        by_category[t.category.name] += t.amount
    
    if by_category:
        top_category, top_amount = max(by_category.items(), key=lambda x: x[1])
        insights.append({
            'type': 'category_leader',
            'category': top_category,
            'amount': top_amount,
            'percent': (top_amount / total_spent) * 100 if total_spent > 0 else 0
        })
    
    # 4. Budget Progress Alerts
    budgets = Budget.query.join(Category).filter(
        Budget.user_id == user_id,
        Category.is_income == False  # Only expense budgets
    ).all()
    
    for budget in budgets:
        spent = sum(
            t.amount for t in transactions 
            if t.category_id == budget.category_id
        )
        
        if budget.limit > 0:  # Only for categories with budgets
            percent = (spent / budget.limit) * 100
            insights.append({
                'type': 'budget_progress',
                'category': budget.category.name,
                'spent': spent,
                'limit': budget.limit,
                'percent': percent
            })
    
    # 5. Month-over-Month Comparison
    prev_month = selected_month - 1 if selected_month > 1 else 12
    prev_year = selected_year if selected_month > 1 else selected_year - 1
    
    current_total = total_spent
    prev_total = db.session.query(
        func.sum(Transaction.amount)
    ).filter(
        Transaction.user_id == user_id,
        Transaction.type == 'debit',
        db.extract('month', Transaction.date) == prev_month,
        db.extract('year', Transaction.date) == prev_year
    ).scalar() or 0
    
    if prev_total > 0:
        change = ((current_total - prev_total) / prev_total) * 100
        insights.append({
            'type': 'trend',
            'direction': 'up' if change > 0 else 'down',
            'percent': abs(round(change, 1)),
            'current_amount': current_total
        })
    
    return insights

def get_monthly_spending(user_id):
    """Returns spending data for last 6 months (only expenses)"""
    now = datetime.now()
    monthly_data = []
    labels = []
    
    for i in range(5, -1, -1):  # Last 6 months including current
        month_start = now.replace(day=1) - timedelta(days=30*i)
        month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        
        monthly_total = db.session.query(
            func.sum(Transaction.amount)
        ).filter(
            Transaction.user_id == user_id,
            Transaction.type == 'debit',  # Only expenses
            Transaction.date >= month_start,
            Transaction.date <= month_end
        ).scalar() or 0
        
        monthly_data.append(abs(float(monthly_total)))
        labels.append(month_start.strftime('%b %Y'))
    
    return {
        'labels': labels,
        'data': monthly_data
    }


def calculate_projections(user_id, current_month, current_year):
    """Calculate expense projections (only expenses)"""
    try:
        now = datetime.now()
        start_of_month = datetime(current_year, current_month, 1)
        days_remaining = max(0, (start_of_month + relativedelta(months=1) - now).days - 1)
        
        projections = {
            'total': 0.0,
            'daily_rate': 0.0,
            'days_remaining': days_remaining,
            'categories': {},
            'last_updated': now.strftime('%Y-%m-%d %H:%M')
        }

        # Get expense categories only
        categories = Category.query.filter(
            ((Category.user_id == user_id) | (Category.is_default == True)),
            Category.is_income == False
        ).all()
        
        # Calculate for each category
        for category in categories:
            # Historical spending (past 3 months)
            historical_spending = db.session.query(
                func.sum(Transaction.amount)
            ).filter(
                Transaction.user_id == user_id,
                Transaction.category_id == category.id,
                Transaction.type == 'debit',  # Only expenses
                Transaction.date >= (start_of_month - relativedelta(months=3)),
                Transaction.date < start_of_month
            ).scalar() or 0.0
            
            # Current month spending
            current_spending = db.session.query(
                func.sum(Transaction.amount)
            ).filter(
                Transaction.user_id == user_id,
                Transaction.category_id == category.id,
                Transaction.type == 'debit',  # Only expenses
                Transaction.date >= start_of_month,
                Transaction.date <= now
            ).scalar() or 0.0
            
            # Calculate daily rate (average over historical period)
            days_in_period = (start_of_month - (start_of_month - relativedelta(months=3))).days
            daily_rate = historical_spending / days_in_period if days_in_period > 0 else 0
            
            projections['categories'][category.name] = {
                'current': current_spending,
                'daily_rate': daily_rate,
                'projected': current_spending + (daily_rate * days_remaining)
            }
        
        return projections
    except Exception as e:
        app.logger.error(f"Projection calculation failed: {str(e)}")
        return {'error': str(e)}

@app.route('/edit-budgets', methods=['GET', 'POST'])
@login_required
def edit_budgets():
    if request.method == 'POST':
        try:
            # Process form submission
            for key, value in request.form.items():
                if key.startswith('budget_'):
                    category_name = key.replace('budget_', '')
                    amount = float(value) if value else 0.0
                    
                    # Skip if amount is 0 or negative
                    if amount <= 0:
                        continue
                    
                    # Find or create category
                    category = Category.query.filter_by(
                        user_id=current_user.id,
                        name=category_name
                    ).first()
                    
                    if not category:
                        # Create new category if it doesn't exist
                        category = Category(
                            user_id=current_user.id,
                            name=category_name,
                            is_default=False
                        )
                        db.session.add(category)
                        db.session.flush()  # Get the category ID
                    
                    # Update or create budget
                    budget = Budget.query.filter_by(
                        user_id=current_user.id,
                        category_id=category.id
                    ).first()
                    
                    if budget:
                        budget.limit = amount
                    else:
                        budget = Budget(
                            user_id=current_user.id,
                            category_id=category.id,
                            limit=amount
                        )
                        db.session.add(budget)
            
            db.session.commit()
            flash('Budgets updated successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating budgets: {str(e)}', 'error')
            return redirect(url_for('edit_budgets'))
    
    # GET request - show form
    clean_slate = request.args.get('clean_slate', False)
    
    if clean_slate:
        budget_dict = {}
    else:
        # Get existing budgets
        budgets = Budget.query.filter_by(user_id=current_user.id).all()
        budget_dict = {b.category.name: b.limit for b in budgets}
        
        # Fill in defaults for missing categories
        default_categories = ['Food', 'Transport', 'Utilities', 
                            'Entertainment', 'Shopping', 'Other']
        for cat in default_categories:
            if cat not in budget_dict:
                budget_dict[cat] = 0.0
    
    return render_template('edit_budgets.html', 
                         budgets=budget_dict,
                         clean_slate=clean_slate)

@app.route('/delete-budget/<category>')
@login_required
def delete_budget(category):
    budget = Budget.query.filter_by(
        user_id=current_user.id,
        category=category
    ).first()
    
    if budget:
        db.session.delete(budget)
        db.session.commit()
        flash(f'{category} budget removed', 'success')
    
    return redirect(url_for('edit_budgets'))

@app.route('/categories', methods=['GET', 'POST'])
@login_required
def manage_categories():
    if request.method == 'POST':
        name = request.form.get('name').strip()
        color = request.form.get('color', '#3B82F6')
        icon = request.form.get('icon', 'tag')
        
        if not name:
            flash('Category name is required', 'error')
            return redirect(url_for('manage_categories'))
        
        # Check for duplicate
        existing = Category.query.filter_by(
            user_id=current_user.id,
            name=name
        ).first()
        
        if existing:
            flash('Category already exists', 'error')
            return redirect(url_for('manage_categories'))
        
        # Create new category
        category = Category(
            user_id=current_user.id,
            name=name,
            color=color,
            icon=icon
        )
        db.session.add(category)
        db.session.commit()
        
        flash('Category added successfully', 'success')
        return redirect(url_for('manage_categories'))
    
    # GET request - show all categories
    categories = Category.query.filter(
        (Category.user_id == current_user.id) | 
        (Category.is_default == True)
    ).order_by(Category.is_default.desc(), Category.name).all()
    
    return render_template('manage_categories.html', categories=categories)

@app.route('/categories/<int:category_id>', methods=['POST'])
@login_required
def update_category(category_id):
    category = Category.query.get_or_404(category_id)
    
    if category.user_id != current_user.id and not category.is_default:
        abort(403)
    
    if request.form.get('_method') == 'DELETE':
        # Check if category is in use
        transaction_count = Transaction.query.filter_by(
            user_id=current_user.id,
            category_id=category.id
        ).count()
        
        if transaction_count > 0:
            flash('Cannot delete category with transactions', 'error')
            return redirect(url_for('manage_categories'))
        
        # Delete category
        db.session.delete(category)
        db.session.commit()
        flash('Category deleted', 'success')
    else:
        # Update category
        if not category.is_default:  # Don't allow editing default categories
            category.name = request.form.get('name', category.name)
            category.color = request.form.get('color', category.color)
            category.icon = request.form.get('icon', category.icon)
            db.session.commit()
            flash('Category updated', 'success')
    
    return redirect(url_for('manage_categories'))

@app.route('/save-budgets', methods=['POST'])
@login_required
def save_budgets():
    # Clear existing budgets
    Budget.query.filter_by(user_id=current_user.id).delete()
    
    # Process submitted budgets
    for key, value in request.form.items():
        if key.startswith('budget_'):
            category_name = key.replace('budget_', '')
            amount = float(value)
            
            if amount > 0:
                # Find or create category
                category = Category.query.filter_by(
                    user_id=current_user.id,
                    name=category_name
                ).first()
                
                if not category:
                    category = Category(
                        user_id=current_user.id,
                        name=category_name,
                        is_default=False
                    )
                    db.session.add(category)
                
                # Create budget
                budget = Budget(
                    user_id=current_user.id,
                    category_id=category.id,
                    limit=amount
                )
                db.session.add(budget)
    
    db.session.commit()
    flash('Budgets saved successfully!', 'success')
    return redirect(url_for('dashboard'))

# Category Selection
@app.route('/select-categories', methods=['GET', 'POST'])
@login_required
def select_categories():
    if request.method == 'POST':
        selected = request.form.getlist('categories')
        
        # Create selected categories
        for cat_id in selected:
            default_cat = Category.query.get(cat_id)
            if default_cat:
                new_cat = Category(
                    user_id=current_user.id,
                    name=default_cat.name,
                    color=default_cat.color,
                    icon=default_cat.icon,
                    is_income=default_cat.is_income
                )
                db.session.add(new_cat)
        
        db.session.commit()
        return redirect(url_for('setup_budget'))
    
    # GET request - show the category selection form
    # Get default categories to show to user
    default_categories = Category.query.filter_by(is_default=True).all()
    
    # If no default categories exist, create some basic ones
    if not default_categories:
        basic_categories = [
            ('Food & Dining', '#FF6B6B', 'utensils', False),
            ('Transportation', '#4ECDC4', 'car', False),
            ('Shopping', '#45B7D1', 'shopping-bag', False),
            ('Bills & Utilities', '#96CEB4', 'file-text', False),
            ('Entertainment', '#FFEAA7', 'film', False),
            ('Other', '#DDA0DD', 'tag', False),
            ('Salary', '#95E1D3', 'dollar-sign', True),
            ('Other Income', '#A8E6CF', 'plus-circle', True)
        ]
        
        for name, color, icon, is_income in basic_categories:
            cat = Category(
                name=name,
                color=color,
                icon=icon,
                is_income=is_income,
                is_default=True,
                user_id=None  # Default categories have no user_id
            )
            db.session.add(cat)
        
        db.session.commit()
        default_categories = Category.query.filter_by(is_default=True).all()
    
    return render_template('select_categories.html', categories=default_categories)


@app.route('/auto-calculate-budgets', methods=['POST'])
@login_required
def auto_calculate_budgets():
    try:
        # Get 3 months of spending history
        three_months_ago = datetime.now() - relativedelta(months=3)
        
        # Calculate average spending per category
        category_spending = db.session.query(
            Category.id,
            Category.name,
            func.avg(Transaction.amount).label('avg_spending')
        ).join(
            Transaction, Transaction.category_id == Category.id
        ).filter(
            Transaction.user_id == current_user.id,
            Transaction.type == 'debit',
            Transaction.date >= three_months_ago
        ).group_by(
            Category.id, Category.name
        ).all()
        
        # Create/update budgets with 20% buffer
        for category_id, name, avg_spending in category_spending:
            if avg_spending > 0:
                budget = Budget.query.filter_by(
                    user_id=current_user.id,
                    category_id=category_id
                ).first()
                
                if not budget:
                    budget = Budget(
                        user_id=current_user.id,
                        category_id=category_id
                    )
                    db.session.add(budget)
                
                budget.limit = avg_spending * 1.2  # 20% buffer
                budget.period = 'monthly'
        
        db.session.commit()
        flash('Budgets automatically calculated with 20% buffer!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error calculating budgets: {str(e)}', 'error')
    
    return redirect(url_for('edit_budgets'))

@app.route('/reports')
@login_required
def reports():
    # Get all unique month/year combinations with transactions
    date_parts = db.session.query(
        db.extract('year', Transaction.date).label('year'),
        db.extract('month', Transaction.date).label('month'),
        func.count(Transaction.id).label('count')
    ).filter(
        Transaction.user_id == current_user.id
    ).group_by(
        db.extract('year', Transaction.date),
        db.extract('month', Transaction.date)
    ).order_by(
        db.extract('year', Transaction.date).desc(),
        db.extract('month', Transaction.date).desc()
    ).all()

    # Group by year
    reports_by_year = defaultdict(list)
    for year, month, count in date_parts:
        reports_by_year[int(year)].append({
            'month': int(month),
            'count': count
        })

    return render_template('reports.html', 
                         reports_by_year=dict(reports_by_year),
                         datetime=datetime)

@app.route('/skip-setup')
@login_required
def skip_setup():
    """Emergency route to skip setup and go directly to dashboard"""
    current_user.has_completed_setup = True
    db.session.commit()
    flash('Setup skipped - you can configure categories and budgets later', 'info')
    return redirect(url_for('dashboard'))

@app.route('/report/<int:year>/<int:month>')
@login_required
def monthly_report(year, month):
    try:
        # Validate input parameters
        if not (1 <= month <= 12):
            flash("Invalid month selected", "error")
            return redirect(url_for('reports'))
        if year < 2000 or year > datetime.now().year + 1:
            flash("Invalid year selected", "error")
            return redirect(url_for('reports'))

        # Calculate date range for the selected month
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        # Calculate previous month for comparison
        if month > 1:
            prev_month = month - 1
            prev_year = year
        else:
            prev_month = 12
            prev_year = year - 1

        # Get all transactions for the selected month
        transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date >= start_date,
            Transaction.date <= end_date
        ).order_by(Transaction.date.desc()).all()

        # Calculate income and expenses
        income = sum(t.amount for t in transactions if t.type == 'credit')
        expenses = sum(t.amount for t in transactions if t.type == 'debit')
        net_change = income - expenses

        # Calculate category breakdown (only expense categories)
        categories = Category.query.filter(
            ((Category.user_id == current_user.id) | (Category.is_default == True)),
            Category.is_income == False
        ).all()

        spending_data = {}
        for category in categories:
            category_spent = sum(
                t.amount for t in transactions 
                if t.category_id == category.id and t.type == 'debit'
            )
            budget = Budget.query.filter_by(
                user_id = current_user.id,
                category_id = category.id
            ).first()
            
            spending_data[category.name] = {
                'spent': category_spent,
                'limit': budget.limit if budget else 0,
                'color': category.color,
                'icon': category.icon
            }

        # Calculate previous month's expenses for comparison
        prev_month_expenses = db.session.query(
            func.sum(Transaction.amount)
        ).filter(
            Transaction.user_id == current_user.id,
            Transaction.type == 'debit',
            db.extract('month', Transaction.date) == prev_month,
            db.extract('year', Transaction.date) == prev_year
        ).scalar() or 0

        spending_change = expenses - prev_month_expenses
        if prev_month_expenses > 0:
            spending_change_percent = (spending_change / prev_month_expenses) * 100
        else:
            spending_change_percent = 0

        # Calculate daily spending pattern
        daily_spending = db.session.query(
            db.extract('day', Transaction.date).label('day'),
            func.sum(Transaction.amount).label('amount')
        ).filter(
            Transaction.user_id == current_user.id,
            Transaction.type == 'debit',
            Transaction.date >= start_date,
            Transaction.date <= end_date
        ).group_by('day').order_by('day').all()

        daily_data = {
            'labels': [str(day) for day in range(1, end_date.day + 1)],
            'data': [0] * end_date.day
        }

        for day, amount in daily_spending:
            if day and amount:
                daily_data['data'][int(day) - 1] = float(amount)

        return render_template('monthly_report.html',
            year=year,
            month=month,
            month_name=start_date.strftime('%B'),
            start_date=start_date,
            end_date=end_date,
            income=income,
            expenses=expenses,
            net_change=net_change,
            spending_data=spending_data,
            transactions=transactions,
            prev_year=prev_year,
            prev_month=prev_month,
            spending_change=spending_change,
            spending_change_percent=abs(round(spending_change_percent, 1)),
            spending_change_direction='up' if spending_change > 0 else 'down',
            daily_data=daily_data,
            datetime=datetime,
            min=min,  # Explicitly pass min function
            abs=abs   # Explicitly pass abs function
        )

    except Exception as e:
        app.logger.error(f"Monthly report error: {str(e)}")
        flash("An error occurred while generating the monthly report", "error")
        return redirect(url_for('reports'))


@app.route('/setup-budget', methods=['GET'])
@login_required
def setup_budget():
    """Show the budget setup method selection page"""
    return render_template('setup_budget.html')

@app.route('/handle-budget-setup', methods=['POST'])
@login_required
def handle_budget_setup():
    """Process the budget method selection"""
    try:
        budget_method = request.form.get('budget_method')
        
        if budget_method == 'auto':
            # Auto budget calculation logic
            three_months_ago = datetime.now() - relativedelta(months=3)
            Budget.query.filter_by(user_id=current_user.id).delete()
            
            category_spending = db.session.query(
                Category.id,
                func.avg(Transaction.amount).label('avg_spending')
            ).join(Transaction).filter(
                Transaction.user_id == current_user.id,
                Transaction.type == 'debit',
                Transaction.date >= three_months_ago
            ).group_by(Category.id).all()
            
            for category_id, avg_spending in category_spending:
                if avg_spending > 0:
                    budget = Budget(
                        user_id=current_user.id,
                        category_id=category_id,
                        limit=avg_spending * 1.2,
                        period='monthly'
                    )
                    db.session.add(budget)
            
            current_user.has_completed_setup = True
            db.session.commit()
            flash('Budgets automatically calculated!', 'success')
            return redirect(url_for('dashboard'))
            
        elif budget_method == 'manual':
            return redirect(url_for('edit_budgets'))
            
        flash('Invalid budget method', 'error')
        return redirect(url_for('setup_budget'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('setup_budget'))


@app.route('/update-transaction-category', methods=['POST'])
@login_required
def update_transaction_category():
    try:
        data = request.get_json()
        transaction = Transaction.query.filter_by(
            id=data['transaction_id'],
            user_id=current_user.id
        ).first_or_404()
        
        category = Category.query.filter_by(
            id=data['new_category_id'],
            user_id=current_user.id
        ).first_or_404()
        
        transaction.category_id = category.id
        db.session.commit()
        
        return jsonify({
            'success': True,
            'new_category_name': category.name,
            'new_category_color': category.color
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400  

if __name__ == '__main__':
    app.run(debug=True)