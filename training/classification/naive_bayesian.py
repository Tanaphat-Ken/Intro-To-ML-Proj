from math import sqrt, pi, exp

class NaiveBayesClassifier:
    def __init__(self, categorical_features, numerical_features, target_feature, use_add1_smoothing=True):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_feature = target_feature
        self.use_add1_smoothing = use_add1_smoothing

        self.classes = []
        self.prior_probs = {} # P(Class)
        self.likelihoods_categorical = {} # P(Feature_cat | Class) with smoothing
        self.numerical_features_stats = {} # {Class: {Feature_num: {'mean': .., 'std': ..}}}
        self.feature_possible_values = {} # For smoothing denominator

    def fit(self, df):
        """
        ฝึกโมเดล Naive Bayes โดยคำนวณ Prior, Likelihoods, และ Stats สำหรับ Numerical Features
        **Likelihoods สำหรับ Categorical Predictors จะไม่ใช้ Add-1 Smoothing**
        """
        self.classes = df[self.target_feature].unique()
        total_samples = len(df)

        # 1. คำนวณ Prior Probabilities
        for cls in self.classes:
            self.prior_probs[cls] = df[df[self.target_feature] == cls].shape[0] / total_samples

        # 2. คำนวณ Likelihoods สำหรับ Categorical Predictors (ไม่มี Smoothing)
        for feature in self.categorical_features:
            self.feature_possible_values[feature] = df[feature].nunique() # ไม่ได้ใช้แล้วถ้าไม่ smoothing
            for cls in self.classes:
                if cls not in self.likelihoods_categorical:
                    self.likelihoods_categorical[cls] = {}
                self.likelihoods_categorical[cls][feature] = {}

                df_class = df[df[self.target_feature] == cls]
                total_count_class = df_class.shape[0]

                for feat_value in df[feature].unique():
                    count_feat_class = df_class[feature].value_counts().get(feat_value, 0)

                    # *** จุดที่แก้ไข 2: ใช้เงื่อนไข use_add1_smoothing ***
                    if self.use_add1_smoothing:
                        prob = (count_feat_class + 1) / (total_count_class + self.feature_possible_values[feature])
                    else:
                        if total_count_class > 0:
                            prob = count_feat_class / total_count_class
                        else:
                            prob = 0.0 # ถ้าไม่มีข้อมูลในคลาสนี้เลย ให้ Likelihood เป็น 0

                    self.likelihoods_categorical[cls][feature][feat_value] = prob

        # 3. คำนวณ Mean และ Standard Deviation สำหรับ Numerical Predictors
        for feature in self.numerical_features:
            for cls in self.classes:
                if cls not in self.numerical_features_stats:
                    self.numerical_features_stats[cls] = {}

                df_class_feature = df[df[self.target_feature] == cls][feature]
                self.numerical_features_stats[cls][feature] = {
                    'mean': df_class_feature.mean(),
                    'std': df_class_feature.std(ddof=1) # ddof=1 for sample std dev
                }
                # Handle cases where std dev is 0 (e.g., all values are same in a class)
                if self.numerical_features_stats[cls][feature]['std'] == 0:
                    self.numerical_features_stats[cls][feature]['std'] = 1e-9 # Prevent division by zero, small epsilon

    def _normal_pdf(self, x, mean, std):
        """Probability Density Function for Normal Distribution."""
        if std == 0: # Should be handled by epsilon in fit, but as a safeguard
            return 1.0 if x == mean else 0.0
        exponent = exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (sqrt(2 * pi) * std)) * exponent

    def _calculate_unnormalized_posterior_with_terms(self, query_features, target_label):
        """
        คำนวณ Numerator (Score) สำหรับคลาสที่ระบุ พร้อมส่งกลับเทอมการคูณแต่ละตัว
        Score = P(X|c) * P(c)
        """
        score = self.prior_probs[target_label]
        terms = [f"{self.prior_probs[target_label]:.4f}"] # เก็บเฉพาะตัวเลขสำหรับส่วน (0.XX * 0.YY...)

        # สำหรับแสดง P(Yes) = ... หรือ P(No) = ... ในบรรทัดแรกของ P(X|c)*P(c)
        detailed_terms_display = [f"P({target_label}) = {self.prior_probs[target_label]:.4f}"]

        for feat_name, feat_value in query_features.items():
            if feat_name in self.categorical_features:
                p_feat_given_label = self.likelihoods_categorical[target_label][feat_name].get(feat_value, 0)
                score *= p_feat_given_label

                display_value = feat_value
                terms.append(f"{p_feat_given_label:.4f}")
                detailed_terms_display.append(f"P({feat_name}={display_value}|{target_label}) = {p_feat_given_label:.4f}")
            elif feat_name in self.numerical_features:
                mean = self.numerical_features_stats[target_label][feat_name]['mean']
                std = self.numerical_features_stats[target_label][feat_name]['std']
                p_feat_given_label = self._normal_pdf(feat_value, mean, std)
                score *= p_feat_given_label
                terms.append(f"{p_feat_given_label:.4f}")
                detailed_terms_display.append(f"PDF({feat_name}={feat_value}|{target_label}) = {p_feat_given_label:.4f}")
            else:
                # This should ideally not happen if features are correctly defined
                print(f"Warning: Feature '{feat_name}' not recognized during prediction. Skipping.")
        return score, terms, detailed_terms_display

    def predict_proba(self, query_features):
        """
        ทำนาย Posterior Probabilities สำหรับแต่ละคลาส
        คืนค่าเป็น dictionary {class: probability}
        """
        scores = {}
        # We don't need detailed terms for predict_proba, only for display
        for cls in self.classes:
            score, _, _ = self._calculate_unnormalized_posterior_with_terms(query_features, cls)
            scores[cls] = score

        total_score = sum(scores.values())

        posterior_probs = {}
        if total_score > 0:
            for cls in self.classes:
                posterior_probs[cls] = scores[cls] / total_score
        else:
            for cls in self.classes:
                posterior_probs[cls] = 0.0
            print("Warning: All unnormalized scores are zero. Posterior probabilities set to 0.0.")

        return posterior_probs

    def predict(self, query_features):
        """
        ทำนายคลาสที่มีความน่าจะเป็นสูงสุด
        """
        posterior_probs = self.predict_proba(query_features)

        if posterior_probs:
            return max(posterior_probs, key=posterior_probs.get)
        else:
            return None