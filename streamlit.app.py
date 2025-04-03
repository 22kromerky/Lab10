import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Water Quality Monitoring Dashboard",
    page_icon="💧",
    layout="wide"
)

# Define helper functions
def load_data(station_file, result_file):
    """Load and preprocess the station and result data"""
    try:
        stations_df = pd.read_csv(station_file)
        results_df = pd.read_csv(result_file)
        
        # Convert date columns to datetime in results_df (adjust column name if needed)
        date_cols = ['ActivityStartDate', 'ActivityEndDate', 'SampleDate', 'ResultDate', 'Date']
        for col in date_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_datetime(results_df[col], errors='coerce')
                break
        
        return stations_df, results_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def get_contaminant_data(results_df):
    """Extract list of contaminants/characteristics from the results data"""
    # Find the characteristic/parameter column
    char_cols = ['CharacteristicName', 'ParameterName', 'Characteristic', 'Parameter', 'ContaminantName']
    char_col = next((col for col in char_cols if col in results_df.columns), None)
    
    if not char_col:
        st.error("Could not find contaminant/characteristic column in results data")
        return [], None
    
    # Get unique contaminants and sort alphabetically
    contaminants = sorted(results_df[char_col].dropna().unique())
    
    return contaminants, char_col

def get_value_data(results_df):
    """Find the result value column in the results data"""
    value_cols = ['ResultValue', 'MeasuredValue', 'Value', 'Result', 'ResultMeasureValue']
    value_col = next((col for col in value_cols if col in results_df.columns), None)
    
    if not value_col:
        st.error("Could not find result value column in results data")
        return None
    
    return value_col

def get_date_column(results_df):
    """Find the date column in the results data"""
    date_cols = ['ActivityStartDate', 'SampleDate', 'ActivityDate', 'Date', 'ResultDate']
    date_col = next((col for col in date_cols if col in results_df.columns 
                    and pd.api.types.is_datetime64_dtype(results_df[col])), None)
    
    if not date_col:
        st.error("Could not find or convert date column in results data")
        return None
    
    return date_col

def get_station_id_columns(stations_df, results_df):
    """Find matching station ID columns in both dataframes"""
    station_id_cols = ['MonitoringLocationIdentifier', 'StationID', 'SiteID', 'LocationID']
    
    station_col = next((col for col in station_id_cols if col in stations_df.columns), None)
    result_col = next((col for col in station_id_cols if col in results_df.columns), None)
    
    if not station_col or not result_col:
        st.error("Could not find matching station ID columns in both datasets")
        return None, None
    
    return station_col, result_col

def filter_data(results_df, stations_df, contaminant, char_col, value_col, date_col, 
                station_col, result_col, value_range, date_range):
    """Filter results data based on user selections"""
    # Filter by contaminant
    filtered_results = results_df[results_df[char_col] == contaminant].copy()
    
    # Convert value column to numeric to ensure proper filtering
    filtered_results[value_col] = pd.to_numeric(filtered_results[value_col], errors='coerce')
    
    # Filter by value range
    filtered_results = filtered_results[
        (filtered_results[value_col] >= value_range[0]) & 
        (filtered_results[value_col] <= value_range[1])
    ]
    
    # Filter by date range
    filtered_results = filtered_results[
        (filtered_results[date_col] >= pd.to_datetime(date_range[0])) &
        (filtered_results[date_col] <= pd.to_datetime(date_range[1]))
    ]
    
    # Join with station data
    filtered_stations = pd.merge(
        filtered_results[[result_col, date_col, value_col]],
        stations_df,
        left_on=result_col,
        right_on=station_col,
        how='inner'
    )
    
    return filtered_results, filtered_stations

def create_map(filtered_stations, station_col, value_col):
    """Create an interactive map of the filtered stations"""
    # Check if we have valid coordinates
    lat_cols = ['LatitudeMeasure', 'Latitude', 'LAT']
    lon_cols = ['LongitudeMeasure', 'Longitude', 'LON']
    
    lat_col = next((col for col in lat_cols if col in filtered_stations.columns), None)
    lon_col = next((col for col in lon_cols if col in filtered_stations.columns), None)
    
    if not lat_col or not lon_col:
        st.error("Could not find latitude/longitude columns in station data")
        return None
    
    # Remove rows with missing coordinates
    map_data = filtered_stations.dropna(subset=[lat_col, lon_col])
    
    if map_data.empty:
        st.warning("No stations with valid coordinates found for the selected criteria")
        return None
    
    # Calculate center coordinates for the map
    center_lat = map_data[lat_col].mean()
    center_lon = map_data[lon_col].mean()
    
    # Create a map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Get unique stations
    unique_stations = map_data.drop_duplicates(subset=[station_col])
    
    # For each station, create a marker
    for idx, row in unique_stations.iterrows():
        # Station data
        station_id = row[station_col]
        station_values = filtered_stations[filtered_stations[station_col] == station_id][value_col]
        
        # Calculate statistics for popup
        max_val = station_values.max()
        min_val = station_values.min()
        avg_val = station_values.mean()
        count = len(station_values)
        
        # Create popup content
        popup_content = f"""
        <b>Station:</b> {row.get('MonitoringLocationName', station_id)}<br>
        <b>ID:</b> {station_id}<br>
        <b>Type:</b> {row.get('MonitoringLocationTypeName', 'N/A')}<br>
        <b>Measurements:</b> {count}<br>
        <b>Min Value:</b> {min_val:.2f}<br>
        <b>Max Value:</b> {max_val:.2f}<br>
        <b>Average Value:</b> {avg_val:.2f}<br>
        """
        
        # Add organization if available
        if 'OrganizationFormalName' in row and pd.notna(row['OrganizationFormalName']):
            popup_content += f"<b>Organization:</b> {row['OrganizationFormalName']}<br>"
        
        # Create popup and tooltip
        popup = folium.Popup(popup_content, max_width=300)
        tooltip = f"{station_id}: Avg {avg_val:.2f}"
        
        # Determine marker color based on average value (higher values = darker)
        # This is a simple red scale, but you could customize based on thresholds
        normalized_value = min(1.0, avg_val / max_val) if max_val > 0 else 0.5
        color = f'#{int(255 - normalized_value * 155):02x}0000'  # Red scale
        
        # Add marker
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=popup,
            tooltip=tooltip,
            icon=folium.Icon(color='red', icon='tint', prefix='fa')
        ).add_to(marker_cluster)
    
    # Add tile layers
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('CartoDB positron').add_to(m)
    folium.LayerControl().add_to(m)
    
    return m

def create_time_series(filtered_results, date_col, value_col, result_col, contaminant):
    """Create a time series plot of contaminant values over time"""
    if filtered_results.empty:
        st.warning("No data available for time series with the selected criteria")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # If there are too many stations, aggregate by date
    unique_stations = filtered_results[result_col].nunique()
    
    if unique_stations > 10:
        # Aggregate by date (weekly average)
        filtered_results['week'] = filtered_results[date_col].dt.to_period('W')
        weekly_data = filtered_results.groupby('week').agg({
            value_col: ['mean', 'min', 'max', 'count']
        }).reset_index()
        weekly_data.columns = ['week', 'mean', 'min', 'max', 'count']
        weekly_data['date'] = weekly_data['week'].dt.to_timestamp()
        
        # Plot aggregated data
        ax.plot(weekly_data['date'], weekly_data['mean'], 'o-', color='#3498db', label='Weekly Average')
        ax.fill_between(weekly_data['date'], weekly_data['min'], weekly_data['max'], 
                        color='#3498db', alpha=0.2, label='Min-Max Range')
        
        # Annotate with sample counts
        for i, row in weekly_data.iterrows():
            if i % max(1, len(weekly_data) // 10) == 0:  # Show only some labels to avoid crowding
                ax.annotate(f"n={row['count']}", 
                           (row['date'], row['mean']),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center',
                           fontsize=8)
        
        ax.set_title(f"{contaminant} - Weekly Aggregated Trend (All Stations)")
    else:
        # Plot individual stations
        for station, group in filtered_results.groupby(result_col):
            group = group.sort_values(by=date_col)
            ax.plot(group[date_col], group[value_col], 'o-', label=f"Station {station}", alpha=0.7)
        
        ax.set_title(f"{contaminant} - Trends by Station")
    
    # Format the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Measured Value')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if there are multiple stations
    if unique_stations > 1 and unique_stations <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis date labels
    fig.autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def get_value_unit(results_df, char_col, value_col, selected_contaminant):
    """Get the unit of measurement for the selected contaminant"""
    unit_cols = ['ResultMeasure/MeasureUnitCode', 'MeasureUnitCode', 'Units', 'Unit']
    unit_col = next((col for col in unit_cols if col in results_df.columns), None)
    
    if not unit_col:
        return ""
    
    # Filter for the selected contaminant
    filtered = results_df[results_df[char_col] == selected_contaminant]
    
    # Get the most common unit
    if not filtered.empty and unit_col in filtered.columns:
        most_common_unit = filtered[unit_col].mode().iloc[0] if not filtered[unit_col].isna().all() else ""
        return most_common_unit
    
    return ""

def main():
    """Main Streamlit application"""
    # App header
    st.title("💧 Water Quality Monitoring Dashboard")
    st.markdown("Upload station and results data to explore contaminant concentrations across monitoring sites.")
    
    # File uploader section
    st.header("📂 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        station_file = st.file_uploader("Upload Station Data (CSV)", type=['csv'])
        
    with col2:
        result_file = st.file_uploader("Upload Results Data (CSV)", type=['csv'])
    
    # Check if both files are uploaded
    if not station_file or not result_file:
        st.info("Please upload both station and results CSV files to continue.")
        
        # Add sample data option
        if st.button("Use Sample Data (Demo Only)"):
            st.warning("This would load sample data in a real deployment. For now, please upload your actual CSV files.")
        
        # Display file format information
        with st.expander("Expected File Format Information"):
            st.markdown("""
            ### Station CSV Expected Columns:
            - **MonitoringLocationIdentifier**: Unique ID for each monitoring station
            - **MonitoringLocationName**: Name of the monitoring location
            - **MonitoringLocationTypeName**: Type of monitoring location
            - **LatitudeMeasure/LongitudeMeasure**: Geographic coordinates
            - **OrganizationFormalName**: Organization managing the station
            
            ### Results CSV Expected Columns:
            - **MonitoringLocationIdentifier**: Station ID (must match station CSV)
            - **CharacteristicName**: Name of the contaminant/parameter measured
            - **ResultValue**: The measured value
            - **ActivityStartDate**: Date of measurement
            - **ResultMeasure/MeasureUnitCode**: Unit of measurement
            
            The app will try to find these columns or similar alternatives in your data.
            """)
            
        return
    
    # Load data
    with st.spinner("Loading and processing data..."):
        stations_df, results_df = load_data(station_file, result_file)
    
    if stations_df is None or results_df is None:
        return
    
    # Display basic data statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Stations", len(stations_df))
    col2.metric("Number of Measurements", len(results_df))
    
    # Get key columns
    contaminants, char_col = get_contaminant_data(results_df)
    value_col = get_value_data(results_df)
    date_col = get_date_column(results_df)
    station_col, result_col = get_station_id_columns(stations_df, results_df)
    
    # Check if we have all required columns
    if not all([contaminants, char_col, value_col, date_col, station_col, result_col]):
        st.error("Could not identify all required columns in the data. Please check the file format.")
        return
    
    col3.metric("Number of Contaminants", len(contaminants))
    
    # Contaminant selection and filters
    st.header("🔍 Select Contaminant and Filters")
    
    # Select contaminant
    selected_contaminant = st.selectbox("Select Contaminant/Parameter", contaminants)
    
    # Filter the data for the selected contaminant to get its range
    contaminant_data = results_df[results_df[char_col] == selected_contaminant]
    # Convert value column to numeric to prevent type errors
    contaminant_values = pd.to_numeric(contaminant_data[value_col], errors='coerce').dropna()
    
    if contaminant_values.empty:
        st.error(f"No numeric values found for {selected_contaminant}")
        return
    
    # Get date range for this contaminant
    contaminant_dates = contaminant_data[date_col].dropna()
    
    if contaminant_dates.empty:
        st.error(f"No valid dates found for {selected_contaminant}")
        return
    
    # Get unit of measurement
    unit = get_value_unit(results_df, char_col, value_col, selected_contaminant)
    unit_display = f" ({unit})" if unit else ""
    
    # Create filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Value range slider
        min_val = float(contaminant_values.min())
        max_val = float(contaminant_values.max())
        
        # If min and max are very close, expand the range slightly
        if abs(max_val - min_val) < 0.001:
            min_val = min_val * 0.9
            max_val = max_val * 1.1
        
        value_range = st.slider(
            f"Value Range{unit_display}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=(max_val - min_val) / 100
        )
    
    with col2:
        # Date range selector
        min_date = contaminant_dates.min().date()
        max_date = contaminant_dates.max().date()
        
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Ensure we have a start and end date
        if len(date_range) < 2:
            date_range = (date_range[0], date_range[0])
    
    # Filter the data based on user selections
    with st.spinner("Filtering data and creating visualizations..."):
        filtered_results, filtered_stations = filter_data(
            results_df, stations_df, selected_contaminant, char_col, value_col, 
            date_col, station_col, result_col, value_range, date_range
        )
    
    if filtered_results.empty or filtered_stations.empty:
        st.warning("No data matches the selected criteria. Try adjusting the filters.")
        return
    
    # Display basic stats about filtered data
    col1, col2, col3 = st.columns(3)
    col1.metric("Matching Measurements", len(filtered_results))
    col2.metric("Matching Stations", filtered_stations[station_col].nunique())
    
    # Calculate statistics
    avg_value = filtered_results[value_col].mean()
    max_value = filtered_results[value_col].max()
    min_value = filtered_results[value_col].min()
    
    col3.metric(f"Average Value{unit_display}", f"{avg_value:.2f}", 
                f"{avg_value - contaminant_values.mean():.2f}")
    
    # Create map
    st.header("🗺️ Monitoring Stations Map")
    station_map = create_map(filtered_stations, station_col, value_col)
    
    if station_map:
        folium_static(station_map, width=1000, height=500)
    
    # Create time series
    st.header("📈 Contaminant Trend Over Time")
    time_series = create_time_series(filtered_results, date_col, value_col, result_col, selected_contaminant)
    
    if time_series:
        st.pyplot(time_series)
    
    # Add download options
    st.header("📊 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download filtered data
        csv = filtered_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"{selected_contaminant}_filtered_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download the plot
        if time_series:
            buf = BytesIO()
            time_series.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="Download Trend Plot (PNG)",
                data=buf,
                file_name=f"{selected_contaminant}_trend_plot.png",
                mime="image/png"
            )
    
    # Data table with the filtered results
    with st.expander("View Filtered Data Table"):
        st.dataframe(filtered_results.sort_values(by=date_col, ascending=False))


# Run the app
if __name__ == "__main__":
    main()
