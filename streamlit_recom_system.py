import streamlit as st
import pandas as pd
import pickle

# functions cần thiết
def get_recommendations(comments_score,hotel_info,reviewer_name,trainset,algorithm,hotels_id,num):
    
    nationality = comments_score[(comments_score['Reviewer Name']==reviewer_name)]['Nationality'].iloc[0]
    reviewer_id_encoded = comments_score[(comments_score['Reviewer Name']==reviewer_name) & (comments_score['Nationality']==nationality)]['reviewer_id_encoded'].iloc[0]
    
    predictions = []
    # Thực hiện dự đoán cho tất cả các item
    for hotel_id in hotels_id:
        # Chuyển đổi hotel_id từ trainset sang raw id
        raw_hotel_id = trainset.to_raw_iid(hotel_id)
        prediction = algorithm.predict(reviewer_id_encoded, raw_hotel_id)
        predictions.append(prediction)

    # Chuyển đổi dự đoán thành DataFrame
    predictions_df = pd.DataFrame({
        'reviewer_id_encoded': [pred.uid for pred in predictions],
        'Hotel ID': [pred.iid for pred in predictions],
        'predicted_score': [pred.est for pred in predictions]
    })

    # Merge với comment_data để lấy user name
    result_df = pd.merge(predictions_df, comments_score[['reviewer_id_encoded','Reviewer ID_new']].drop_duplicates(), left_on='reviewer_id_encoded', right_on='reviewer_id_encoded', how='left')

    # Merge với info để lấy hotel name
    result_df = pd.merge(result_df, hotel_info, left_on='Hotel ID', right_on='Hotel_ID', how='left')

    #Lấy danh sách những hotel mà user đó đã ở
    hotel_list = comments_score[comments_score['reviewer_id_encoded']==reviewer_id_encoded]['Hotel ID'].tolist()

    #Loại khỏi kết quả những hotel đã ở
    result_df = result_df[~result_df['Hotel ID'].isin(hotel_list)]
    result_df = result_df.dropna(subset=['Hotel_Name'])

    # Sắp xếp kết quả theo điểm số dự đoán giảm dần và lấy 5 gợi ý hàng đầu
    result_df = result_df.sort_values(by='predicted_score', ascending=False)


    # In ra 5 gợi ý
    # return result_df[['Reviewer ID_new', 'Hotel_Name', 'predicted_score']].iloc[:num+1]
    return result_df[['Hotel_Name','Hotel_ID','Hotel_Description','Hotel_Address','Total_Score','Hotel_Rank','predicted_score']].iloc[:num]

hotels_id_pkl = "hotels_id.pkl"
algo_pkl = "surprise_model_algo.pkl"
trainset_pkl = "surprise_model_trainset.pkl"

comments = "comments_score.csv"
info = "hotel_info.csv"
df_comments = pd.read_csv(comments)
df_info = pd.read_csv(info)

# Open and read file to surprise model
with open(hotels_id_pkl, 'rb') as f:
    hotels_id_new = pickle.load(f)
with open(algo_pkl, 'rb') as f:
    algo_new = pickle.load(f)
with open(trainset_pkl, 'rb') as f:
    trainset_new = pickle.load(f)
    
# test = get_recommendations(df_comments,df_info,reviewer_name='Abhishek',trainset=trainset_new,algorithm=algo_new,hotels_id=hotels_id_new, num=5)
# print(test)

# Hiển thị đề xuất ra bảng
def display_recommended_hotels(recommended_hotels, cols):
    for i in range(0, len(recommended_hotels), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:   
                    st.write(hotel['Hotel_Name'])                    
                    expander = st.expander(f"Description")
                    hotel_description = hotel['Hotel_Description']
                    truncated_description = ' '.join(hotel_description.split()[:80]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           


# Đọc dữ liệu khách sạn
df_hotels = pd.read_csv(info)
# Lấy 10 khách sạn
random_hotels = df_hotels.head(n=10)
# print(random_hotels)

# Using menu
st.title("""DATA SCIENCE AND MACHINE LEARNING COURSE \n
         Trung Tâm Tin Học Đại Học Khoa Học Tự Nhiên""")
menu = ["Home", "Capstone Project", "Hotel Recommendation System"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Khóa học](https://csc.edu.vn/data-science-machine-learning/data-science-and-machine-learning-certificate-version-2024_285)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""CHỦ ĐỀ 2: HOTEL RECOMMENDATION SYSTEM - COLLABORARIVE ANALYSIS \n
    HỌC VIÊN: LÝ NGỌC TƯỜNG MINH \n
    NỘI DUNG: GỢI Ý CHO KHÁCH CÁC KHÁCH SẠN DỰA TRÊN LỊCH SỬ ĐÁNH GIÁ CỦA KHÁCH \n
    HƯỚNG DẪN SỬ DỤNG: NHẬP TÊN VÀ QUÓC TỊCH ĐỂ LẤY CÁC KHÁCH SẠN GỢI Ý""")
elif choice == 'Hotel Recommendation System':
        
        
    st.session_state.random_hotels = random_hotels


    ###### Giao diện Streamlit ######
    st.image('hotel.jpg', use_column_width=True)

    # Dropdown for selecting the user's name
    selected_name = st.selectbox("Chọn tên:",
                                options=df_comments['Reviewer Name'].unique())

    # Dropdown for selecting the user's nationality
    selected_nationality = st.selectbox("Chọn quốc tịch:",
                                        options=df_comments[df_comments['Reviewer Name'] == selected_name]['Nationality'].unique())

    # Find the corresponding user_id
    user_id = df_comments[(df_comments['Reviewer Name'] == selected_name) & (df_comments['Nationality'] == selected_nationality)]['reviewer_id_encoded'].values[0]

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_user_name' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_user_name = None
        
    if 'recommendations' not in st.session_state:
    # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.recommendations = None

    # Display the selected hotel
    st.write(f"Bạn đã chọn: {selected_name} - {selected_nationality} - {user_id}")

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_user_name = selected_name
    st.session_state.selected_user_nationality = selected_nationality

    if st.session_state.selected_user_name:
        st.write("Tên: ", selected_name)
        st.write("Quốc tịch: ", selected_nationality)
        st.write("User ID: ", user_id)
        
        # Display the top 5 hotel recommendations
        st.write('##### Các khách sạn khác bạn có thể quan tâm:')
        recommendations = get_recommendations(df_comments,df_info,reviewer_name=st.session_state.selected_user_name,trainset=trainset_new,algorithm=algo_new,hotels_id=hotels_id_new, num=5)
        # display_recommended_hotels(recommendations,cols=3)
        
        st.session_state.recommendations = recommendations
        # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.recommendations.iterrows()]
        # Tạo một dropdown với options là các tuple này
        selected_hotel = st.radio(
            "Chọn khách sạn",
            options=hotel_options,
            format_func=lambda x: x[0]  # Hiển thị tên khách sạn
        )
        
        # Display the selected hotel
        st.write("Bạn đã chọn:", selected_hotel)

        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_hotel_id = selected_hotel[1]
        if st.session_state.selected_hotel_id:
            st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
            # Hiển thị thông tin khách sạn được chọn
            hotel_info = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]
            # hotel_info = df_hotels[df_hotels['Hotel_ID'] == selected_hotel[1]]
            
            hotel_description = hotel_info['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('Thông tin:')
            st.write(truncated_description, '...')
            st.write("Địa chỉ:", hotel_info['Hotel_Address'].values[0])
            st.write("Điểm số:", hotel_info['Total_Score'].values[0])
            st.write("Ranking:", hotel_info['Hotel_Rank'].values[0])
        
    else:
        st.write(f"Không tìm thấy khách lưu trú với tên {st.session_state.selected_user_name} và quốc tịch {st.session_state.selected_user_nationality}")
