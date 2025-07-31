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
from main_parser import run_parser_and_cleaner
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
            return redirect(url_for('setup_budget'))
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
    # Calculate income vs expenses
    income = db.session.query(
        func.sum(Transaction.amount)
    ).filter(
        Transaction.user_id == current_user.id,
        Transaction.type == 'credit'
    ).scalar() or 0

    expenses = db.session.query(
        func.sum(Transaction.amount)
    ).filter(
        Transaction.user_id == current_user.id,
        Transaction.type == 'debit'
    ).scalar() or 0

    net_worth = income - expenses

    # Get all categories for the current user (including default ones)
    categories = Category.query.filter(
        (Category.user_id == current_user.id) | 
        (Category.is_default == True)
    ).all()
    
    # Get all transactions for the current user
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    # Get all budgets for the current user
    budgets = {b.category_id: b for b in Budget.query.filter_by(user_id=current_user.id).all()}
    
    # Calculate spending per category
    spending_data = {}
    total_spent = 0
    category_totals = defaultdict(float)
    
    for category in categories:
        # Calculate spent amount for this category
        category_spent = sum(
            t.amount for t in transactions 
            if t.category_id == category.id and t.type == 'debit'
        )
        
        # Get budget limit for this category (0 if no budget set)
        budget_limit = budgets.get(category.id, Budget(limit=0)).limit
        
        spending_data[category.name] = {
            'spent': category_spent,
            'limit': budget_limit,
            'remaining': max(0, budget_limit - category_spent),
            'color': category.color,
            'icon': category.icon,
            'is_income': category.is_income
        }
        
        total_spent += category_spent
        category_totals[category.name] = category_spent
    
    # Calculate budget utilization percentage
    if budgets:
        budget_utilization = sum(
            min(100, (data['spent'] / data['limit']) * 100) 
            for data in spending_data.values() if data['limit'] > 0
        ) / len(budgets)
    else:
        budget_utilization = 0
    
    # Find top spending category
    top_category = ""
    top_category_amount = 0
    if category_totals:
        top_category = max(category_totals, key=category_totals.get)
        top_category_amount = category_totals[top_category]
    
    # Get monthly spending data
    monthly_spending = get_monthly_spending(current_user.id)
    
    # Generate insights
    insights = generate_insights(current_user.id)
    
    # Calculate projections
    projections = calculate_projections(current_user.id)
    
    # Get recent transactions (last 5)
    recent_transactions = Transaction.query.filter_by(
        user_id=current_user.id
    ).order_by(Transaction.date.desc()).limit(5).all()
    
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
        datetime=datetime,
        timedelta=timedelta,
        monthly_spending=monthly_spending,
        abs=abs,
        projections=projections,
        categories=categories
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

@app.route('/setup-budget', methods=['GET', 'POST'])
def setup_budget():
    if request.method == 'POST':
        budget_type = request.form.get('budget_type')
        
        if budget_type == 'manual':
            # Start with empty categories for manual setup
            return redirect(url_for('edit_budgets', clean_slate=True))
        
        elif budget_type == 'auto':
            auto_calculate_budgets(current_user.id)
            flash('Budgets created automatically!', 'success')
        else:  # default
            create_default_budgets(current_user.id)
            flash('Default budgets set!', 'success')
        
        return redirect(url_for('dashboard'))
    
    return render_template('budget_setup.html')

def auto_calculate_budgets(user_id):
    # Add your calculation logic here
    pass

def create_default_budgets(user_id):
    # Add default budget categories
    pass

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

def generate_insights(user_id):
    transactions = Transaction.query.filter_by(user_id=user_id).all()
    budgets = Budget.query.filter_by(user_id=user_id).all()
    
    insights = []
    
    # 1. Top Spending Category (Card)
    if transactions:
        by_category = {}
        for t in transactions:
            by_category[t.category] = by_category.get(t.category, 0) + t.amount
        
        top_category = max(by_category.items(), key=lambda x: x[1])
        insights.append({
            'type': 'category_leader',
            'category': top_category[0],
            'amount': top_category[1],
            'icon': '💰'  # Emoji or font-awesome class
        })
    
    # 2. Budget Progress (Progress Bar)
    for budget in budgets:
        spent = sum(t.amount for t in transactions if t.category == budget.category)
        if spent > 0:
            insights.append({
                'type': 'budget_progress',
                'category': budget.category,
                'spent': spent,
                'limit': budget.limit,
                'percent': min(100, (spent / budget.limit) * 100)
            })
    
    # 3. Spending Trend (Comparison)
    if len(transactions) >= 2:
        current_month = datetime.now().month
        last_month = current_month - 1 if current_month > 1 else 12
        
        current_total = sum(t.amount for t in transactions if t.date.month == current_month)
        last_total = sum(t.amount for t in transactions if t.date.month == last_month)
        
        if current_total > 0 and last_total > 0:
            change = ((current_total - last_total) / last_total) * 100
            insights.append({
                'type': 'trend',
                'direction': 'up' if change > 0 else 'down',
                'percent': abs(round(change)),
                'current_amount': current_total
            })
    
    return insights

def get_monthly_spending(user_id):
    """Returns spending data for the last 6 months"""
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
            Transaction.date >= month_start,
            Transaction.date <= month_end
        ).scalar() or 0
        
        monthly_data.append(abs(float(monthly_total)))
        labels.append(month_start.strftime('%b %Y'))
    
    return {
        'labels': labels,
        'data': monthly_data
    }


def calculate_projections(user_id):
    """Calculate projected spending with enhanced error handling"""
    try:
        now = datetime.now()
        start_of_month = now.replace(day=1)
        days_remaining = max(0, (now.replace(day=1) + relativedelta(months=1) - now).days - 1)
        
        # Initialize default structure
        projections = {
            'total': 0.0,
            'daily_rate': 0.0,
            'days_remaining': days_remaining,
            'categories': {},
            'last_updated': now.strftime('%Y-%m-%d %H:%M')
        }

        # Get all transactions (optimized single query)
        transactions = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.date >= (now - relativedelta(months=3)).replace(day=1)
        ).all()

        if not transactions:
            return projections

        # Calculate daily rates per category
        monthly_totals = defaultdict(lambda: defaultdict(float))
        for t in transactions:
            month_key = t.date.strftime('%Y-%m')
            monthly_totals[month_key][t.category] += t.amount

        # Process each category
        current_month_spending = defaultdict(float)
        for t in Transaction.query.filter_by(user_id=user_id).filter(Transaction.date >= start_of_month).all():
            current_month_spending[t.category] += t.amount

        for category in set(current_month_spending.keys()).union(
                       *[set(m.keys()) for m in monthly_totals.values()]):
            daily_rates = []
            for month, categories in monthly_totals.items():
                if category in categories:
                    month_date = datetime.strptime(month, '%Y-%m').date()
                    days_in_month = (month_date.replace(day=28) + timedelta(days=4)).day
                    daily_rates.append(categories[category] / days_in_month)
            
            if daily_rates:
                median_rate = statistics.median(daily_rates)
                projections['categories'][category] = {
                    'current': current_month_spending.get(category, 0.0),
                    'daily_rate': median_rate,
                    'projected': current_month_spending.get(category, 0.0) + (median_rate * days_remaining)
                }

        # Calculate total projection
        if monthly_totals:
            total_daily_rates = [
                sum(categories.values()) / ((datetime.strptime(month, '%Y-%m').replace(day=28) + timedelta(days=4)).day)
                for month, categories in monthly_totals.items()
            ]
            projections['daily_rate'] = statistics.median(total_daily_rates)
            projections['total'] = sum(current_month_spending.values()) + (projections['daily_rate'] * days_remaining)

        return projections

    except Exception as e:
        app.logger.error(f"Projection calculation failed: {str(e)}")
        return {
            'error': str(e),
            'total': 0.0,
            'daily_rate': 0.0,
            'days_remaining': max(0, (datetime.now().replace(day=1) + relativedelta(months=1) - datetime.now()).days - 1),
            'categories': {},
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
        }

@app.route('/edit-budgets', methods=['GET', 'POST'])
@login_required
def edit_budgets():
    if request.method == 'POST':
        # Process form submission
        for key, value in request.form.items():
            if key.startswith('budget_'):
                category_name = key.replace('budget_', '')
                amount = float(value) if value else 0.0
                
                # Find or create category
                category = Category.query.filter_by(
                    user_id=current_user.id,
                    name=category_name
                ).first()
                
                if not category and amount > 0:
                    category = Category(
                        user_id=current_user.id,
                        name=category_name,
                        is_default=False
                    )
                    db.session.add(category)
                    db.session.flush()  # Get the category ID
                
                # Update or create budget
                if amount > 0:
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
                else:
                    # Remove budget if amount is 0
                    Budget.query.filter_by(
                        user_id=current_user.id,
                        category_id=category.id
                    ).delete()
        
        db.session.commit()
        flash('Budgets updated successfully!', 'success')
        return redirect(url_for('dashboard'))

    # GET request - show form
    clean_slate = request.args.get('clean_slate', False)
    
    if clean_slate:
        # Start with empty dictionary for clean slate
        budget_dict = {}
    else:
        # Get existing budgets
        budgets = Budget.query.filter_by(user_id=current_user.id).all()
        budget_dict = {b.category.name: b.limit for b in budgets}
        
        # Fill in defaults for missing categories (only if not clean slate)
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
    
    # Show default categories for selection
    default_categories = Category.query.filter_by(is_default=True).all()
    return render_template('select_categories.html', categories=default_categories)

def calculate_auto_budgets(user_id):
    # Get 3 months of transaction history
    three_months_ago = datetime.now() - timedelta(days=90)
    
    # Calculate average spending per category
    results = db.session.query(
        Category.name,
        func.avg(Transaction.amount).label('avg_spending')
    ).join(
        Transaction, Transaction.category_id == Category.id
    ).filter(
        Transaction.user_id == user_id,
        Transaction.date >= three_months_ago,
        Transaction.type == 'debit'  # Only consider expenses
    ).group_by(
        Category.name
    ).all()
    
    budgets = {}
    for category, avg_spending in results:
        # Add 20% buffer to average spending
        budgets[category] = avg_spending * 1.2
    
    return budgets

if __name__ == '__main__':
    app.run(debug=True)