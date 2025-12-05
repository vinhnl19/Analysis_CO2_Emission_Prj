from datetime import datetime
import pandas as pd
import os
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import yaml
import time
from ga_optimizer import ga_optimize_changes
from es_optimizer import es_optimize_changes
from de_optimizer import de_optimizer_changes

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

def input_co2_target(df, country):
    country_data = df[df['Country'] == country].sort_values('Year')
    print(f"\nLịch sử phát thải CO2 của {country}: ")
    for _, row in country_data.iterrows():
        print(f"Năm {int(row['Year'])}: {row['Co2_MtCO2']:.2f} MtCO2")
    
    while True:
        target = input("Nhập CO2 target cho năm sau: ")
        try:
            return float(target)
        except ValueError:
            print("Nhập không hợp lệ, thử lại.")

def select_changeable_features(features, seq_data, country_name):
    print("\nDanh sách các feature: ")
    for i,f in enumerate(features):
        print(f"{i+1}. {f}")
    raw = input("\nBạn muốn được tham vấn giá trị của những feature nào?\n"
                "Hãy nhập số thứ tự, mỗi số cách nhau dấu phẩy (,): ")
    
    chosen_idx = []
    for x in raw.split(","):
        x = x.strip()
        if x.isdigit() and 1 <= int(x) <= len(features):
            chosen_idx.append(int(x)-1)

    chosen_features = [features[i] for i in chosen_idx]
    not_chosen_features = [f for f in features if f not in chosen_features]

    selected_features = []
    fixed_features = {}

    print("\n=== NHẬP GIÁ TRỊ CHO CÁC FEATURE KHÔNG ĐƯỢC THAM VẤN ===")

    lastest_row = seq_data.iloc[-1]

    for f in not_chosen_features:
        current_value = lastest_row[f]
        print(f"\n- {f} cua {country_name} vao nam gan nhat hien tai la: {current_value}")

        use_default = input("Ban co muon su dung gia tri nay cho nam sau khong? (y/n): ").strip().lower()

        if use_default == "y":
            next_val = current_value
        else:
            while True:
                user_val = input(f"Gia tri chi dinh cua {f} trong nam sau cua ban la: ").strip()
                try:
                    next_val = float(user_val)
                    break
                except:
                    print("Gia tri khong hop le, hay nhap so!")

        fixed_features[f] = next_val
    
    print("\n=== NHẬP GIÁ TRỊ BIÊN CHO CÁC FEATURE ĐƯỢC THAM VẤN ===")

    for f in chosen_features:
        while True:
            raw = input(f"Nhập giá trị biên (% giảm tối đa, % tăng tối đa) cho {f} (vd: -30, 50): ")
            parts = raw.split(",")
            if len(parts) != 2:
                print("Hãy nhập hai số cách nhau dấu phẩy.")
                continue

            try:
                min_pct = float(parts[0].strip())
                max_pct = float(parts[1].strip())
                break
            except:
                print("Không hợp lệ, hãy nhập dạng số.")

        selected_features.append({
            "feature": f,
            "min_pct": min_pct,
            "max_pct": max_pct
        })

    return selected_features, fixed_features



def predict_co2(model, le, scaler, country_name, sequence_data_features):
    num_feature = int(sequence_data_features.shape[1])

    seq_df = pd.DataFrame(sequence_data_features, columns=FEATURE)
    seq_df_log = np.log1p(seq_df)
    seq_scaled = scaler.transform(seq_df_log)
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
    y_pred_real = np.expm1(y_pred_real)
    return y_pred_real



def main():
    # 1. Load data, model, scaler, le và cost guide
    df = pd.read_csv("../Data/filled_data.csv")    

    model_path = os.path.join(os.path.dirname(__file__), "../Model/Final/best_model_gru3_final.keras")
    scaler_path = os.path.join(os.path.dirname(__file__), "../Model/Final/scaler_quantile.save")
    le_path = os.path.join(os.path.dirname(__file__), "../Model/Final/labelencoder_country.save")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    model_xgb = joblib.load("./XGBoost/Model_XGBoost.joblib")

    countries = df['Country'].unique().tolist()

    while True:
        print("===================================================================")
        print("Chọn flow cần chạy:")
        print("1. Dự đoán CO2 tương lai cho một quốc gia")
        print("2. Khuyến nghị giá trị feature để đạt CO2 target")
        print("3. Dự đoán CO2 dựa trên các giá trị feature của quốc gia trong năm")
        print("4. Thoát")
        choice = input("Chọn: ").strip()
        if choice == '1':
            country = select_country(countries)
            year = select_year()
            seq_data = load_sequence_data(df, country, year)
            seq_features = seq_data[FEATURE].to_numpy()
            predicted_co2 = predict_co2(model, le, scaler, country_name=country, sequence_data_features=seq_features)
            print(f"\nPredicted CO2: {predicted_co2:.2f}")
        elif choice == '2':
            country = select_country(countries)
            co2_target = input_co2_target(df=df, country=country)

            latest_year = int(df[df['Country'] == country]['Year'].max())
            seq_data = load_sequence_data(df, country, target_year=latest_year + 1)

            feature_selection, fixed_features = select_changeable_features(FEATURE_CORE, seq_data=seq_data, country_name=country)

            def predict_fn(indiv_changes, fixed_features):
                x_values = {}
                x_full = []

                for f in FEATURE_CORE:
                    if f in indiv_changes:
                        original_val = seq_data[f].to_numpy()[-1]
                        pct = indiv_changes[f] / 100.0
                        new_val = original_val * (1 + pct)
                        x_values[f] = new_val
                        x_full.append(new_val)
                    else:
                        new_val = fixed_features[f]
                        x_values[f] = new_val
                        x_full.append(new_val)
                
                x_df_scale = pd.DataFrame([x_full], columns=FEATURE_CORE)

                pred = model_xgb.predict(x_df_scale)[0]

                return pred, x_values
            
            print("Dang kiem tra...", end="", flush=True)
            start = time.time()
            best_change, best_fitness, best_predicted_co2, best_x = es_optimize_changes(
                                                                        feature_selection=feature_selection, 
                                                                        fixed_features=fixed_features,
                                                                        predict_fn=predict_fn, 
                                                                        co2_target=co2_target)
            end = time.time()
            elapsed = end - start

            print(feature_selection)
            print(fixed_features)

            print("\rCo ket qua roi!           ") 
            print(f"Thời gian thực thi: {elapsed:.4f} giây")
            print(f"Giá trị % cần thay đổi: {best_change} ")
            print(f"Fitness của lời giải: {best_fitness} ")
            print(f"Giá trị CO2 đạt được từ lời giải: {best_predicted_co2} ")
            print("Giá trị của lời giải:")
            for f, v in best_x.items():
                print(f"  {f}: {round(v, 2)}")
        elif choice == '3':
            country = select_country(countries)
            print("\nNhập giá trị cho từng feature (theo đơn vị hiện có):")
            user_features = []
            for f in FEATURE_CORE:
                while True:
                    val = input(f"{f}: ")
                    try:
                        val = float(val)
                        user_features.append(val)
                        break
                    except ValueError:
                        print("Giá trị không hợp lệ, hãy nhập số.")

            x_df = pd.DataFrame([user_features], columns=FEATURE_CORE)
            predicted_co2 = model_xgb.predict(x_df)[0]

            print(f"\nPredicted CO2 với giá trị nhập vào: {predicted_co2:.2f}")

        elif choice == '4':
            print("Thoát chương trình!")
            break
        else:
            print("Lựa chọn không hợp lệ, thử lại.")





    # # 2. User chọn country & year
    # country = select_country(countries)
    # year = select_year()

    # # 3. Load sequence data
    # seq_data = load_sequence_data(df, country, year)
    # # Giả lập: tách feature và CO2 cho predict
    # seq_features = seq_data[FEATURE].to_numpy()

    # # 4. Predict CO2
    # predicted_co2 = predict_co2(model, le, scaler, country_name=country, sequence_data_features=seq_features)
    # print(f"\nPredicted CO2: {predicted_co2}")

    # # 5. User nhập CO2 target
    # co2_target = input_co2_target()

    # # 6. User chọn feature có thể thay đổi + cost
    # feature_selection = select_changeable_features(FEATURE_CORE, cost_guide)

    # # 7. Recommendation
    # def predict_fn(indiv_changes):

    #     x = seq_data[FEATURE_CORE].to_numpy().copy()[-1].copy()
    #     for f, pct in indiv_changes.items():
    #         idx = FEATURE_CORE.index(f)
    #         x[idx] *= (1 + pct/100.0)

    #     x_df_scale = pd.DataFrame([x], columns=FEATURE_CORE)

    #     pred = model_xgb.predict(x_df_scale)[0]
    #     return pred
    
    # print("Dang kiem tra...", end="", flush=True)
    # start = time.time()
    # best_change, best_fitness, best_predicted_co2 = de_optimizer_changes(feature_selection=feature_selection, predict_fn=predict_fn, predicted_co2=predicted_co2, co2_target=co2_target)
    # end = time.time()
    # elapsed = end - start

    # print("\rCo ket qua roi!           ") 
    # print(f"Thời gian thực thi: {elapsed:.4f} giây")
    # print(best_change)
    # print(best_fitness)
    # print(best_predicted_co2)
    
if __name__ == "__main__":
    main()


