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
        # Validate file extension
        if not file.filename.lower().endswith(('.csv', '.pdf')):
            flash('Only CSV and PDF files are allowed', 'error')
            return redirect(url_for('transactions'))

        # Save file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        # Process based on file type
        if filename.lower().endswith('.pdf'):
            temp_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'processed.csv')
            run_parser_and_cleaner(temp_path, temp_csv)
            processor = TransactionProcessor()
            count = processor.process_uploaded_file(current_user.id, temp_csv)
            os.remove(temp_csv)
        else:
            processor = TransactionProcessor()
            count = processor.process_uploaded_file(current_user.id, temp_path)
        
        os.remove(temp_path)
        flash(f'Successfully imported {count} transactions!', 'success')
        
    except Exception as e:
        app.logger.error(f"Upload failed: {str(e)}")
        flash(f'Error processing file: {str(e)}', 'error')
        # Clean up temp files if they exist
        for temp_file in [temp_path, temp_csv]:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    return redirect(url_for('transactions'))




import joblib
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from models import db, Transaction, Category

class TransactionProcessor:
    def __init__(self):
        try:
            self.model = joblib.load('transaction_classifier.pkl')
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def _parse_date(self, date_str):
        """Parse date string into date object"""
        try:
            return pd.to_datetime(date_str).date()
        except Exception as e:
            raise ValueError(f"Invalid date format: {date_str} - {str(e)}")

    def _parse_amount(self, amount_str):
        """Parse amount string into float"""
        try:
            amount_str = str(amount_str).replace('J$', '').replace(',', '').strip()
            return float(amount_str)
        except Exception as e:
            raise ValueError(f"Invalid amount: {amount_str} - {str(e)}")

    def _predict_category(self, description, amount, category_map, default_category_id):
        """Predict category for a transaction using ML model"""
        try:
            if not self.model:
                return default_category_id
                
            input_data = pd.DataFrame([{
                'Description': str(description).lower(),
                'Amount': abs(amount),
                'Type': 'DEBIT' if amount < 0 else 'CREDIT'
            }])
            
            predicted_name = self.model.predict(input_data)[0].lower()
            return category_map.get(predicted_name, default_category_id)
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return default_category_id

    def process_uploaded_file(self, user_id, csv_path):
        """Process uploaded transaction file and save to database"""
        try:
            # Read CSV with error handling
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                raise ValueError(f"Invalid CSV file: {str(e)}")

            # Validate required columns
            required_columns = {'Date', 'Description', 'Amount', 'Type'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {', '.join(missing)}")

            transactions = []
            
            # Get user's available categories
            user_categories = Category.query.filter(
                (Category.user_id == user_id) | 
                (Category.is_default == True)
            ).all()
            
            # Create mapping of category names to IDs
            category_map = {cat.name.lower(): cat.id for cat in user_categories}
            default_category_id = next(
                (cat.id for cat in user_categories if cat.name.lower() == 'other'),
                user_categories[0].id if user_categories else None
            )

            for _, row in df.iterrows():
                try:
                    # Parse and validate transaction data
                    date = self._parse_date(row['Date'])
                    amount = self._parse_amount(row['Amount'])
                    description = str(row['Description']).strip()
                    
                    if not description or amount == 0:
                        continue

                    # Predict category using ML model
                    category_id = self._predict_category(
                        description=description,
                        amount=amount,
                        category_map=category_map,
                        default_category_id=default_category_id
                    )

                    transactions.append({
                        'user_id': user_id,
                        'category_id': category_id,
                        'date': date,
                        'description': description,
                        'amount': -abs(amount) if str(row['Type']).strip().upper() == 'DEBIT' else abs(amount),
                        'type': 'debit' if amount < 0 else 'credit'
                    })

                except Exception as e:
                    print(f"Skipping row {_}: {str(e)}")
                    continue

            # Bulk insert valid transactions
            if transactions:
                db.session.bulk_insert_mappings(Transaction, transactions)
                db.session.commit()

            return len(transactions)

        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Processing failed: {str(e)}")

    def process_single_transaction(self, user_id, transaction_data):
        """Process a single transaction (for manual entry)"""
        try:
            # Get user's categories
            user_categories = Category.query.filter(
                (Category.user_id == user_id) | 
                (Category.is_default == True)
            ).all()
            
            category_map = {cat.name.lower(): cat.id for cat in user_categories}
            default_category_id = next(
                (cat.id for cat in user_categories if cat.name.lower() == 'other'),
                user_categories[0].id if user_categories else None
            )

            # Predict category
            category_id = self._predict_category(
                description=transaction_data['description'],
                amount=transaction_data['amount'],
                category_map=category_map,
                default_category_id=default_category_id
            )

            # Create transaction
            transaction = Transaction(
                user_id=user_id,
                category_id=category_id,
                date=transaction_data['date'],
                description=transaction_data['description'],
                amount=transaction_data['amount'],
                type='debit' if transaction_data['amount'] > 0 else 'credit'
            )

            db.session.add(transaction)
            db.session.commit()
            return transaction

        except Exception as e:
            db.session.rollback()
            raise ValueError(f"Failed to process transaction: {str(e)}")
