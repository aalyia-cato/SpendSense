import joblib
from models import db, Category

class CategoryPredictor:
    def __init__(self, model_path='transaction_classifier.pkl'):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            print(f"⚠️ Failed to load model: {str(e)}")
            self.model = None

    def predict_for_user(self, user_id, description, amount, is_income=False):
        if not self.model:
            return self._get_default_category(user_id, is_income)

        try:
            # Prepare input with amount and type
            input_text = f"{description.lower()} {amount} {'credit' if is_income else 'debit'}"

            # Get prediction
            prediction = self.model.predict([input_text])[0]

            # Find matching category
            user_categories = Category.query.filter(
                (Category.user_id == user_id),
                (Category.is_income == is_income)
            ).all()

            # Try exact match first
            for cat in user_categories:
                if prediction.lower() == cat.name.lower():
                    return cat.id

            # Try partial match
            for cat in user_categories:
                if prediction.lower() in cat.name.lower() or cat.name.lower() in prediction.lower():
                    return cat.id

            return self._get_default_category(user_id, is_income)

        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return self._get_default_category(user_id, is_income)

    def _get_default_category(self, user_id, is_income):
        default_name = 'Other Income' if is_income else 'Other'
        default = Category.query.filter(
            (Category.user_id == user_id),
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