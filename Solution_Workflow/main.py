from datetime import datetime
import pandas as pd
import os
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import yaml
from ga_optimizer import ga_optimize_changes

FEATURE = ['Co2_MtCO2',
    'Population',
    'GDP',
    'Industry_on_GDP',
    'Government_Expenditure_on_Education',
    'Global_Climate_Risk_Index',
    'HDI',
    'Renewable_Energy_Percent',
    'Deforest_Percent',
    'Energy_Capita_kWh']
FEATURE_CORE = [
    'Population',
    'GDP',
    'Industry_on_GDP',
    'Government_Expenditure_on_Education',
    'Global_Climate_Risk_Index',
    'HDI',
    'Renewable_Energy_Percent',
    'Deforest_Percent',
    'Energy_Capita_kWh']

def select_country(countries):
    print("Danh sách quốc gia:")
    for i, c in enumerate(countries):
        print(f"{i+1}. {c}")
    while True:
        choice = input("Chọn quốc gia (nhập số): ")
        if choice.isdigit() and 1 <= int(choice) <= len(countries):
            return countries[int(choice)-1]
        else:
            print("Lựa chọn không hợp lệ, thử lại.")

def select_year():
    current_year = datetime.now().year
    max_year = current_year + 1
    while True:
        year = input(f"Nhập năm cần dự đoán CO2 (<= {max_year}): ")
        if year.isdigit() and int(year) <= max_year:
            return int(year)
        else:
            print("Năm không hợp lệ, thử lại.")


def load_sequence_data(df, country, target_year, seq_len=3):
    # Lấy dữ liệu seq_len năm trước target_year
    years = [target_year - i - 1 for i in reversed(range(seq_len))]
    seq_data = df[(df['Country'] == country) & (df['Year'].isin(years))].sort_values('Year')
    
    if len(seq_data) < seq_len:
        print("Không đủ dữ liệu sequence, chỉ sử dụng dữ liệu hiện có.")
    
    print("\nSequence data:")
    print(seq_data)
    return seq_data

def input_co2_target():
    while True:
        target = input("Nhập CO2 target cho năm này: ")
        try:
            return float(target)
        except ValueError:
            print("Nhập không hợp lệ, thử lại.")

def select_changeable_features(features, cost_guide):
    selected = {}
    print("\nChọn feature có thể thay đổi và cost (1-5):")
    for f in features:
        allow = input(f"Feature {f} có thể thay đổi? (y/n): ").strip().lower()
        if allow == 'y':
            while True:
                cost = input(f"Chọn cost level (1-5) cho {f}: ")
                if cost.isdigit() and 1 <= int(cost) <= 5:
                    max_change = cost_guide["cost_levels"][int(cost)]["max_change_pct"]
                    selected[f] = {"cost": int(cost), "max_change_pct": max_change}
                    break
                else:
                    print("Cost không hợp lệ.")
    return selected

def predict_co2(model, le, scaler, country_name, sequence_data_features):
    num_feature = int(sequence_data_features.shape[1])

    seq_df = pd.DataFrame(sequence_data_features, columns=FEATURE)
    seq_scaled = scaler.transform(seq_df)
    X_new = np.expand_dims(seq_scaled, axis=0)

    if country_name not in le.classes_:
        print(f"Canh bao: '{country_name}' chua co trong encoder, dung code 0 mac dinh")
        country_code = 0
    else:
        country_code = le.transform([country_name])[0]

    X_country = np.array([[country_code]], dtype='int32')

    y_pred_scaled = model.predict([X_new, X_country], verbose=0)
    y_pred_real = scaler.inverse_transform(
        np.concatenate([y_pred_scaled, np.zeros((1, num_feature - 1))], axis=1)
    )[0, 0]
    return y_pred_real

def recommend_feature_changes(predicted_co2, co2_target, feature_selection):
    diff = predicted_co2 - co2_target
    print(f"\nPredicted CO2: {predicted_co2:.2f}, Target: {co2_target:.2f}, Difference: {diff:.2f}")
    
    print("\nGợi ý thay đổi feature:")
    # Demo simple: giảm % theo max_change, chia đều
    num_features = len(feature_selection)
    for f, info in feature_selection.items():
        # ví dụ giảm tỉ lệ % tối đa scaled theo difference
        change_pct = min(info["max_change_pct"], abs(diff)/num_features)
        direction = "giảm" if diff > 0 else "tăng"
        print(f"Feature {f}: {direction} khoảng {change_pct:.1f}% (cost={info['cost']})")


def main():
    # 1. Load data, model, scaler, le và cost guide
    df = pd.read_csv("../Data/filled_data.csv")
    with open("cost_level_guide.yaml") as f:
        cost_guide = yaml.safe_load(f)
    

    model_path = os.path.join(os.path.dirname(__file__), "../Model/best_model_gru3.keras")
    scaler_path = os.path.join(os.path.dirname(__file__), "../Model/scaler_minmax.save")
    le_path = os.path.join(os.path.dirname(__file__), "../Model/labelencoder_country.save")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    countries = df['Country'].unique().tolist()

    # 2. User chọn country & year
    country = select_country(countries)
    year = select_year()

    # 3. Load sequence data
    seq_data = load_sequence_data(df, country, year)
    # Giả lập: tách feature và CO2 cho predict
    seq_features = seq_data[FEATURE].to_numpy()

    # 4. Predict CO2
    predicted_co2 = predict_co2(model, le, scaler, country_name=country, sequence_data_features=seq_features)
    print(f"\nPredicted CO2: {predicted_co2}")

    # 5. User nhập CO2 target
    co2_target = input_co2_target()

    # 6. User chọn feature có thể thay đổi + cost
    feature_selection = select_changeable_features(FEATURE_CORE, cost_guide)

    # 7. Recommendation
    def predict_fn(indiv_changes):
        model_rf = joblib.load("./Predict_Linear/co2_model_rf.joblib")
        scaler_x = joblib.load("./Predict_Linear/scaler_x.joblib")

        x = seq_data[FEATURE_CORE].to_numpy().copy()[-1].copy()
        for f, pct in indiv_changes.items():
            idx = FEATURE_CORE.index(f)
            x[idx] *= (1 + pct/100.0)

        x_df_scale = pd.DataFrame([x], columns=FEATURE_CORE)

        x_scaled = scaler_x.transform(x_df_scale)
        pred = model_rf.predict(x_scaled)[0]
        return pred
        # seq_mod = seq_features.copy()

        # for f, pct in indiv_changes.items():
        #     idx = FEATURE.index(f)
        #     seq_mod[-1, idx] *= (1 + pct / 100.0)

        # return predict_co2(model, le, scaler, country, seq_mod)
    print("Dang kiem tra...", end="", flush=True)
    best_change, best_fitness, best_predicted_co2 = ga_optimize_changes(feature_selection=feature_selection, predict_fn=predict_fn, predicted_co2=predicted_co2, co2_target=co2_target)

    print("\rCo ket qua roi!           ") 
    print(best_change)
    print(best_fitness)
    print(best_predicted_co2)
    
    # recommend_feature_changes(predicted_co2, co2_target, feature_selection)

if __name__ == "__main__":
    main()


