import joblib
import pandas as pd
from models import db, Category

class CategoryPredictor:
    def __init__(self, model_path='transaction_classifier.pkl'):
        try:
            self.model = joblib.load(model_path)
            # Also load the label encoder that was used during training
            self.label_encoder = joblib.load('label_encoder.pkl')  # You'll need to save this
        except Exception as e:
            print(f"⚠️ Failed to load model: {str(e)}")
            self.model = None
            self.label_encoder = None

    def predict_for_user(self, user_id, description, amount, is_income=False):
        if not self.model or not self.label_encoder:
            return self._get_default_category(user_id, is_income)

        try:
            # Create DataFrame in the EXACT format your model expects
            # Based on your training data: Description (lowercase), Amount (float), Type (uppercase)
            input_data = pd.DataFrame({
                'Description': [description.lower().strip()],
                'Amount': [float(amount)],
                'Type': ['CREDIT' if is_income else 'DEBIT']  # Match training data format
            })
            
            print(f"🔍 Model input: {input_data.to_dict('records')[0]}")

            # Get prediction (this returns encoded labels)
            prediction_encoded = self.model.predict(input_data)[0]
            
            # Decode the prediction back to category name
            predicted_category = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            print(f"🎯 Predicted category: '{predicted_category}' for '{description[:50]}...'")

            # Find matching category for this user
            user_categories = Category.query.filter(
                (Category.user_id == user_id) | (Category.is_default == True)
            ).all()

            print(f"📋 Available categories: {[cat.name for cat in user_categories]}")

            # Try exact match first
            for cat in user_categories:
                if predicted_category.lower() == cat.name.lower():
                    print(f"✅ Exact match found: {cat.name}")
                    return cat.id

            # Try partial match
            for cat in user_categories:
                if (predicted_category.lower() in cat.name.lower() or 
                    cat.name.lower() in predicted_category.lower()):
                    print(f"✅ Partial match found: {cat.name} (predicted: {predicted_category})")
                    return cat.id

            # If no match, create the predicted category for this user
            print(f"🆕 Creating new category: {predicted_category}")
            return self._create_predicted_category(user_id, predicted_category, is_income)

        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return self._get_default_category(user_id, is_income)

    def _create_predicted_category(self, user_id, category_name, is_income):
        """Create a new category based on model prediction"""
        try:
            new_category = Category(
                user_id=user_id,
                name=category_name,
                color='#3B82F6',  # Default blue
                icon='tag',
                is_income=is_income
            )
            db.session.add(new_category)
            db.session.commit()
            return new_category.id
        except Exception as e:
            print(f"⚠️ Failed to create predicted category: {str(e)}")
            return self._get_default_category(user_id, is_income)

    def _get_default_category(self, user_id, is_income):
        default_name = 'Other Income' if is_income else 'Other'
        default = Category.query.filter(
            (Category.user_id == user_id) | (Category.is_default == True),
            (Category.name.ilike(default_name))
        ).first()

        if not default:
            try:
                default = Category(
                    user_id=user_id,
                    name=default_name,
                    color='#808080',
                    icon='tag',
                    is_income=is_income
                )
                db.session.add(default)
                db.session.commit()
            except Exception as e:
                print(f"⚠️ Failed to create default category: {str(e)}")
                return None

        return default.id if default else None