from flask import Flask, jsonify, request, send_file, make_response
import pandas as pd
import folium
import os
from final_xg import predict_risk
from html import escape
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Load pre-generated datasets (prefer H2 predictions)
# -----------------------------
DATASET_CANDIDATES = [
    "dataset/risk_2025_H2_weekly_predictions.csv",
    "dataset/risk_2025_weekly_predictions.csv",
    "dataset/risk_2025_predictions.csv",
]

df = None
dataset_path = None
for p in DATASET_CANDIDATES:
    if os.path.exists(p):
        dataset_path = p
        df = pd.read_csv(p)
        break

if df is None:
    df = pd.DataFrame(columns=[
        "city","zone","latitude","longitude","year","month","week",
        "temperature","humidity","rainfall","pop_density","previous_incidents",
        "population_density","sanitation_score","risk_score"
    ])

# Helpers
def get_color(score: float) -> str:
    """Legacy fixed color buckets; kept for compatibility."""
    if score < 30:
        return "green"
    elif score < 70:
        return "orange"
    else:
        return "red"

def risk_to_color(score: float) -> str:
    """Return a hex color on a green->yellow->red gradient for 0..100 risk."""
    # Normalize 0..100
    s = max(0.0, min(100.0, float(score))) / 100.0
    # Green (0, 176, 80) to Yellow (255, 204, 0) to Red (220, 38, 38)
    if s <= 0.5:
        # Green -> Yellow
        t = s / 0.5
        r = int((1 - t) * 0 + t * 255)
        g = int((1 - t) * 176 + t * 204)
        b = int((1 - t) * 80 + t * 0)
    else:
        # Yellow -> Red
        t = (s - 0.5) / 0.5
        r = int((1 - t) * 255 + t * 220)
        g = int((1 - t) * 204 + t * 38)
        b = int((1 - t) * 0 + t * 38)
    return f"#{r:02x}{g:02x}{b:02x}"

def get_filtered_df(city: str, zone: str | None, month: int | None, week: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    q = df.copy()
    if city:
        q = q[q["city"] == city]
    if zone and zone.lower() != "all":
        q = q[q["zone"] == zone]
    if month:
        q = q[q["month"] == int(month)]
    if week:
        q = q[q["week"] == int(week)]
    return q

def create_map(city: str, zone: str | None, month: int | None, week: int | None = None, layer: str = "markers") -> str:
    city_df = get_filtered_df(city, zone, month, week)
    if city_df.empty:
        raise ValueError("No data for the selected filters.")

    m = folium.Map(
        location=[city_df["latitude"].mean(), city_df["longitude"].mean()],
        zoom_start=10,
        control_scale=True
    )
    # Remove default white background/border from Leaflet's DivIcon globally on this map
    from folium import Element
    css = Element(
        """
        <style>
        .leaflet-div-icon { background: transparent !important; border: none !important; }
        .sugg-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
        .sugg-chip {
            display: inline-block;
            padding: 4px 8px;
            font-size: 12px;
            line-height: 1.2;
            border-radius: 12px;
            background: rgba(255,255,255,0.9);
            color: #111827;
            border: 1px solid rgba(17,24,39,0.15);
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
            white-space: nowrap;
        }
        .sugg-title { margin-top: 6px; font-weight: 600; }
        </style>
        """
    )
    m.get_root().html.add_child(css)

    # Suggestion helper based on risk and local drivers
    def zone_suggestions_html(row: pd.Series) -> str:
        risk = float(row.get("risk_score", 0))
        temp = float(row.get("temperature", 0))
        hum = float(row.get("humidity", 0))
        rain = float(row.get("rainfall", 0))
        pop = float(row.get("population_density", row.get("pop_density", 0)))
        sani = float(row.get("sanitation_score", 75))

        items = []
        # Base by risk
        if risk >= 70:
            items.append("Immediate fogging and source reduction in the next 24–48 hours.")
            items.append("Door-to-door inspection and larval habitat elimination (weekly).")
            items.append("Intensify case surveillance and set up fever clinics.")
        elif risk >= 30:
            items.append("Weekly cleanup drives and targeted larviciding in hotspots.")
            items.append("Community awareness on dry-day and container management.")
            items.append("Ward-level monitoring with rapid response teams.")
        else:
            items.append("Maintain routine surveillance and periodic IEC activities.")

        # Feature-driven modifiers
        if rain > 200:
            items.append("Clear stagnant water and de-silt stormwater drains within 48 hours (high rainfall).")
        if 28 <= temp <= 34 and 60 <= hum <= 85:
            items.append("Vector-conducive climate detected; intensify larval source control this week.")
        if hum > 80:
            items.append("Prioritize indoor inspections and eliminate damp breeding spots (high humidity).")
        if pop > 10000:
            items.append("Deploy additional ward teams and micro-plan by street clusters (high density).")
        if sani < 60:
            items.append("Improve solid-waste removal and sanitation services in low-score pockets.")

        chips = "".join(f"<span class='sugg-chip'>{escape(str(t))}</span>" for t in items)
        return f"<div class='sugg-row'>{chips}</div>"

    if layer == "markers":
        from folium.features import DivIcon
        show_label = (isinstance(zone, str) and zone.lower() != "all")
        selected_week = int(week) if week is not None else None
        for _, row in city_df.iterrows():
            risk = float(row["risk_score"])
            color = risk_to_color(risk)
            # Slightly bigger circles: 12..24 px
            radius = max(12, min(24, 12 + (risk / 100.0) * 12))
            # Circle with gradient color
            display_week = selected_week if selected_week is not None else int(row['week'])
            popup_html = (
                f"{row['zone']}<br><b>Risk:</b> {risk:.1f}"
                f"<br><b>Month:</b> {int(row['month'])}"
                f"<br><b>Week:</b> {display_week}"
                f"<div class='sugg-title'>Suggestions:</div>" + zone_suggestions_html(row)
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,  # kept for popup hover halo
                weight=0,
                stroke=False,  # avoid ring/border; show solid fill only
                fill=True,
                fill_color=color,
                fill_opacity=0.90,
                popup=popup_html,
            ).add_to(m)
            # Only show numeric label inside the marker when a specific zone is selected
            if show_label:
                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    icon=DivIcon(
                        icon_size=(0, 0),
                        icon_anchor=(0, 0),
                        class_name="risk-label",
                        html=(
                            f"<div style='transform: translate(-50%, -50%);"
                            f" color:black; font-weight:800; font-size:14px; line-height:1;"
                            f" -webkit-text-stroke: 1.2px rgba(255,255,255,0.95);"
                            f" text-shadow: 0 0 2px rgba(255,255,255,0.9), 0 0 6px rgba(255,255,255,0.6);"
                            f" pointer-events:none;'>"
                            f"{int(round(risk))}</div>"
                        ),
                    ),
                ).add_to(m)
    else:
        try:
            from folium.plugins import HeatMap
            heat_data = city_df[["latitude", "longitude", "risk_score"]].values.tolist()
            HeatMap(heat_data, radius=17, blur=20, min_opacity=0.35, max_zoom=10).add_to(m)
        except Exception:
            for _, row in city_df.iterrows():
                risk = float(row["risk_score"])
                color = get_color(risk)
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f"{row['zone']}<br>Risk: {risk:.1f}"
                          f"<br>Month: {int(row['month'])}"
                          f"<br>Week: {int(row['week'])}"
                ).add_to(m)

    map_path = "dataset/_ui_filtered_map.html"
    # Ensure output directory exists before saving the map
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    m.save(map_path)
    return map_path

# -----------------------------
# API for UI options and summaries
# -----------------------------
@app.route("/api/cities")
def api_cities():
    if df is None or df.empty:
        return jsonify([])
    cities = sorted(df["city"].dropna().unique().tolist())
    return jsonify(cities)

@app.route("/api/zones")
def api_zones():
    city = request.args.get("city", type=str)
    if not city or df is None or df.empty:
        return jsonify([])
    zones = sorted(df[df["city"] == city]["zone"].dropna().unique().tolist())
    return jsonify(["All"] + zones)

@app.route("/api/months")
def api_months():
    return jsonify([7, 8, 9, 10, 11, 12])

@app.route("/api/weeks")
def api_weeks():
    city = request.args.get("city", type=str)
    zone = request.args.get("zone", type=str)
    month = request.args.get("month", type=int)
    q = get_filtered_df(city, zone, month, week=None)
    if q is None or q.empty:
        return jsonify([])
    weeks = sorted(q["week"].dropna().unique().tolist())
    return jsonify(weeks)

@app.route("/api/summary")
def api_summary():
    city = request.args.get("city", type=str)
    zone = request.args.get("zone", type=str)
    month = request.args.get("month", type=int)
    week = request.args.get("week", type=int)
    q = get_filtered_df(city, zone, month, week)
    if q is None or q.empty:
        return jsonify({"low": 0, "medium": 0, "high": 0})
    low = int((q["risk_score"] < 30).sum())
    med = int(((q["risk_score"] >= 30) & (q["risk_score"] < 70)).sum())
    high = int((q["risk_score"] >= 70).sum())
    return jsonify({"low": low, "medium": med, "high": high})

@app.route("/api/download")
def api_download():
    city = request.args.get("city", type=str)
    zone = request.args.get("zone", type=str)
    month = request.args.get("month", type=int)
    week = request.args.get("week", type=int)
    q = get_filtered_df(city, zone, month, week)
    if q is None or q.empty:
        return "No data", 404
    path = "dataset/_filtered_export.csv"
    q.to_csv(path, index=False)
    return send_file(path, as_attachment=True)

@app.route("/api/trend")
def api_trend():
    city = request.args.get("city", type=str)
    zone = request.args.get("zone", type=str)
    if df is None or df.empty:
        return jsonify([])
    
    q = df.copy()
    if city:
        q = q[q["city"] == city]
    if zone and zone.lower() != "all":
        q = q[q["zone"] == zone]
    
    # Group by month and week, calculate average risk
    trend_data = q.groupby(['month', 'week']).agg({
        'risk_score': ['mean', 'count']
    }).round(2)
    
    trend_data.columns = ['avg_risk', 'zone_count']
    trend_data = trend_data.reset_index()
    
    # Create trend points
    trend_points = []
    for _, row in trend_data.iterrows():
        month_names = {7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        trend_points.append({
            'period': f"{month_names.get(int(row['month']), row['month'])} W{int(row['week'])}",
            'month': int(row['month']),
            'week': int(row['week']),
            'avg_risk': float(row['avg_risk']),
            'zone_count': int(row['zone_count'])
        })
    
    return jsonify(trend_points)

# -----------------------------
# Hospital Alert System API
# -----------------------------
@app.route("/api/hospitals")
def api_hospitals():
    # Mock hospital data - in real implementation, this would come from a database
    hospitals = [
        {"id": 1, "name": "Apollo Hospital", "city": "Chennai", "zone": "Anna Nagar", "contact": "+91-44-2829-3333", "email": "emergency@apollochennai.com"},
        {"id": 2, "name": "Fortis Malar Hospital", "city": "Chennai", "zone": "Adyar", "contact": "+91-44-4289-4289", "email": "info@fortismalar.com"},
        {"id": 3, "name": "MIOT International", "city": "Chennai", "zone": "Manapakkam", "contact": "+91-44-4200-1000", "email": "info@miotinternational.com"},
        {"id": 4, "name": "Government General Hospital", "city": "Chennai", "zone": "Park Town", "contact": "+91-44-2819-3000", "email": "ggh@tn.gov.in"},
        {"id": 5, "name": "Sri Ramachandra Medical Centre", "city": "Chennai", "zone": "Porur", "contact": "+91-44-4528-1000", "email": "info@sriramachandra.edu.in"},
        {"id": 6, "name": "Coimbatore Medical College Hospital", "city": "Coimbatore", "zone": "Race Course", "contact": "+91-422-259-1100", "email": "cmch@tn.gov.in"},
        {"id": 7, "name": "PSG Hospitals", "city": "Coimbatore", "zone": "Peelamedu", "contact": "+91-422-257-1170", "email": "info@psghospitals.com"},
        {"id": 8, "name": "Kovai Medical Center", "city": "Coimbatore", "zone": "Avinashi Road", "contact": "+91-422-4324-324", "email": "info@kmchhospitals.com"}
    ]
    
    city = request.args.get("city", type=str)
    zone = request.args.get("zone", type=str)
    
    filtered_hospitals = hospitals
    if city:
        filtered_hospitals = [h for h in filtered_hospitals if h["city"] == city]
    if zone and zone.lower() != "all":
        filtered_hospitals = [h for h in filtered_hospitals if h["zone"] == zone]
    
    return jsonify(filtered_hospitals)

@app.route("/api/send-alert", methods=["POST"])
def send_alert():
    data = request.get_json()
    alert_type = data.get("alert_type", "medium")
    city = data.get("city", "")
    zone = data.get("zone", "")
    risk_score = data.get("risk_score", 0)
    hospitals = data.get("hospitals", [])
    message = data.get("message", "")
    
    # In a real implementation, this would send actual emails/SMS
    # For now, we'll just return a success response
    alert_data = {
        "status": "success",
        "message": f"Alert sent to {len(hospitals)} hospitals",
        "alert_details": {
            "type": alert_type,
            "city": city,
            "zone": zone,
            "risk_score": risk_score,
            "timestamp": "2025-09-25 15:53:00",
            "hospitals_notified": len(hospitals)
        }
    }
    
    return jsonify(alert_data)

# -----------------------------
# Government Reporting System API
# -----------------------------
@app.route("/api/generate-report")
def generate_report():
    report_type = request.args.get("type", "summary")
    city = request.args.get("city", "")
    start_date = request.args.get("start_date", "2025-07-01")
    end_date = request.args.get("end_date", "2025-12-31")
    
    if df is None or df.empty:
        return jsonify({"error": "No data available"})
    
    # Filter data
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df["city"] == city]
    
    # Generate comprehensive report data
    report_data = {
        "report_metadata": {
            "generated_at": "2025-09-25 16:00:00",
            "report_type": report_type,
            "period": f"{start_date} to {end_date}",
            "total_zones_analyzed": len(filtered_df),
            "cities_covered": filtered_df["city"].nunique() if not filtered_df.empty else 0
        },
        "executive_summary": {
            "overall_risk_level": "MEDIUM",
            "high_risk_zones": len(filtered_df[filtered_df["risk_score"] >= 70]) if not filtered_df.empty else 0,
            "medium_risk_zones": len(filtered_df[(filtered_df["risk_score"] >= 30) & (filtered_df["risk_score"] < 70)]) if not filtered_df.empty else 0,
            "low_risk_zones": len(filtered_df[filtered_df["risk_score"] < 30]) if not filtered_df.empty else 0,
            "average_risk_score": round(filtered_df["risk_score"].mean(), 2) if not filtered_df.empty else 0
        },
        "city_breakdown": [],
        "zone_details": [],
        "recommendations": [
            "Increase vector control activities in high-risk zones",
            "Enhance hospital preparedness in medium-risk areas",
            "Continue routine surveillance in low-risk zones",
            "Focus on community awareness programs",
            "Strengthen inter-departmental coordination"
        ],
        "action_items": [
            {"priority": "HIGH", "action": "Deploy additional medical teams to high-risk zones", "deadline": "Within 48 hours"},
            {"priority": "MEDIUM", "action": "Conduct community awareness campaigns", "deadline": "Within 1 week"},
            {"priority": "LOW", "action": "Review and update response protocols", "deadline": "Within 1 month"}
        ]
    }
    
    # City-wise breakdown
    if not filtered_df.empty:
        city_stats = filtered_df.groupby('city').agg({
            'risk_score': ['mean', 'max', 'min', 'count']
        }).round(2)
        
        for city_name in city_stats.index:
            city_data = {
                "city": city_name,
                "avg_risk": float(city_stats.loc[city_name, ('risk_score', 'mean')]),
                "max_risk": float(city_stats.loc[city_name, ('risk_score', 'max')]),
                "min_risk": float(city_stats.loc[city_name, ('risk_score', 'min')]),
                "total_zones": int(city_stats.loc[city_name, ('risk_score', 'count')]),
                "risk_level": "HIGH" if city_stats.loc[city_name, ('risk_score', 'mean')] >= 70 else "MEDIUM" if city_stats.loc[city_name, ('risk_score', 'mean')] >= 30 else "LOW"
            }
            report_data["city_breakdown"].append(city_data)
        
        # Top 10 highest risk zones
        top_risk_zones = filtered_df.nlargest(10, 'risk_score')[['city', 'zone', 'risk_score', 'month', 'week']]
        for _, zone in top_risk_zones.iterrows():
            report_data["zone_details"].append({
                "city": zone['city'],
                "zone": zone['zone'],
                "risk_score": float(zone['risk_score']),
                "period": f"Month {int(zone['month'])}, Week {int(zone['week'])}",
                "status": "CRITICAL" if zone['risk_score'] >= 80 else "HIGH" if zone['risk_score'] >= 70 else "MEDIUM"
            })
    
    return jsonify(report_data)

@app.route("/api/email-report", methods=["POST"])
def email_report():
    data = request.get_json()
    recipients = data.get("recipients", [])
    report_type = data.get("report_type", "summary")
    city = data.get("city", "")
    
    # In real implementation, this would send actual emails
    # For now, simulate email sending
    email_result = {
        "status": "success",
        "message": f"Report sent to {len(recipients)} recipients",
        "details": {
            "report_type": report_type,
            "city": city if city else "All Cities",
            "recipients_count": len(recipients),
            "sent_at": "2025-09-25 16:00:00"
        }
    }
    
    return jsonify(email_result)

# -----------------------------
# Predictive Analytics API
# -----------------------------
@app.route("/api/predictions")
def get_predictions():
    city = request.args.get("city", "")
    weeks_ahead = int(request.args.get("weeks", 4))
    
    # Mock prediction data - in real implementation, this would use ML models
    predictions = []
    base_risk = 45
    
    for week in range(1, weeks_ahead + 1):
        # Simulate seasonal increase
        seasonal_factor = 1 + (week * 0.1)
        predicted_risk = min(100, base_risk * seasonal_factor + (week * 2))
        
        predictions.append({
            "week": week,
            "period": f"Week +{week}",
            "predicted_risk": round(predicted_risk, 1),
            "confidence": round(95 - (week * 2), 1),
            "trend": "INCREASING" if predicted_risk > base_risk else "STABLE",
            "factors": ["Monsoon season", "Population density", "Previous outbreaks"]
        })
    
    return jsonify({
        "city": city if city else "All Cities",
        "predictions": predictions,
        "model_info": {
            "algorithm": "XGBoost Ensemble",
            "last_trained": "2025-09-20",
            "accuracy": "95.2%"
        }
    })

# -----------------------------
# Resource Management API
# -----------------------------
@app.route("/api/resources")
def get_resources():
    city = request.args.get("city", "")
    
    # Mock resource data
    resources = {
        "medical_resources": {
            "hospitals": 8 if city == "Chennai" else 3 if city == "Coimbatore" else 15,
            "icu_beds": 120 if city == "Chennai" else 45 if city == "Coimbatore" else 200,
            "isolation_wards": 15 if city == "Chennai" else 8 if city == "Coimbatore" else 25,
            "medical_staff": 450 if city == "Chennai" else 180 if city == "Coimbatore" else 750
        },
        "vector_control": {
            "fogging_machines": 25 if city == "Chennai" else 12 if city == "Coimbatore" else 45,
            "field_teams": 18 if city == "Chennai" else 8 if city == "Coimbatore" else 30,
            "surveillance_points": 85 if city == "Chennai" else 35 if city == "Coimbatore" else 150
        },
        "emergency_supplies": {
            "rapid_test_kits": 5000 if city == "Chennai" else 2000 if city == "Coimbatore" else 8000,
            "iv_fluids": 1200 if city == "Chennai" else 500 if city == "Coimbatore" else 2000,
            "medications": 850 if city == "Chennai" else 350 if city == "Coimbatore" else 1500
        },
        "utilization_rates": {
            "hospital_occupancy": "68%",
            "staff_deployment": "85%",
            "equipment_usage": "72%"
        }
    }
    
    return jsonify(resources)

# -----------------------------
# Real-time Notifications API
# -----------------------------
@app.route("/api/notifications")
def get_notifications():
    user_role = request.args.get("role", "admin")
    
    # Mock notification data - in real implementation, this would come from a database
    notifications = [
        {
            "id": 1,
            "type": "critical",
            "title": "High Risk Alert - Chennai Anna Nagar",
            "message": "Risk score has exceeded 85% in Anna Nagar zone. Immediate action required.",
            "timestamp": "2025-09-25 18:30:00",
            "read": False,
            "priority": "HIGH",
            "city": "Chennai",
            "zone": "Anna Nagar"
        },
        {
            "id": 2,
            "type": "warning",
            "title": "Medium Risk Alert - Coimbatore Peelamedu",
            "message": "Risk score increased to 65% in Peelamedu zone. Enhanced monitoring recommended.",
            "timestamp": "2025-09-25 17:45:00",
            "read": False,
            "priority": "MEDIUM",
            "city": "Coimbatore",
            "zone": "Peelamedu"
        },
        {
            "id": 3,
            "type": "info",
            "title": "Weekly Report Generated",
            "message": "Tamil Nadu weekly dengue risk report has been generated and sent to government officials.",
            "timestamp": "2025-09-25 16:00:00",
            "read": True,
            "priority": "LOW",
            "city": "All",
            "zone": "All"
        },
        {
            "id": 4,
            "type": "success",
            "title": "Hospital Alert Sent Successfully",
            "message": "Alert sent to 5 hospitals in Chennai regarding increased risk levels.",
            "timestamp": "2025-09-25 15:30:00",
            "read": True,
            "priority": "MEDIUM",
            "city": "Chennai",
            "zone": "Multiple"
        }
    ]
    
    # Filter by role if needed
    if user_role == "hospital":
        notifications = [n for n in notifications if n["type"] in ["critical", "warning"]]
    elif user_role == "municipality":
        notifications = [n for n in notifications if n["city"] != "All"]
    
    return jsonify(notifications)

@app.route("/api/notifications/<int:notification_id>/read", methods=["POST"])
def mark_notification_read(notification_id):
    # In real implementation, this would update the database
    return jsonify({"status": "success", "message": f"Notification {notification_id} marked as read"})

@app.route("/api/notifications/mark-all-read", methods=["POST"])
def mark_all_notifications_read():
    # In real implementation, this would update all notifications for the user
    return jsonify({"status": "success", "message": "All notifications marked as read"})

# -----------------------------
# PDF Report Generation
# -----------------------------
def generate_pdf_report(report_data):
    """Generate a PDF report from the report data"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1e40af')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#374151')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        textColor=colors.HexColor('#374151')
    )
    
    # Title
    story.append(Paragraph("Dengue Risk Intelligence Report", title_style))
    story.append(Paragraph(f"Tamil Nadu State - Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Zones Analyzed', str(report_data['report_metadata']['total_zones_analyzed'])],
        ['Cities Covered', str(report_data['report_metadata']['cities_covered'])],
        ['Average Risk Score', f"{report_data['executive_summary']['average_risk_score']}%"],
        ['High Risk Zones', str(report_data['executive_summary']['high_risk_zones'])],
        ['Medium Risk Zones', str(report_data['executive_summary']['medium_risk_zones'])],
        ['Low Risk Zones', str(report_data['executive_summary']['low_risk_zones'])]
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # City Breakdown
    if report_data.get('city_breakdown'):
        story.append(Paragraph("City-wise Risk Analysis", heading_style))
        city_data = [['City', 'Total Zones', 'Risk Level', 'Average Risk Score']]
        for city in report_data['city_breakdown']:
            city_data.append([
                city['city'],
                str(city['total_zones']),
                city['risk_level'],
                f"{city['avg_risk']}%"
            ])
        
        city_table = Table(city_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        city_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
        ]))
        story.append(city_table)
        story.append(Spacer(1, 20))
    
    # High Risk Zones
    if report_data.get('zone_details'):
        story.append(Paragraph("High Risk Zones (Top 10)", heading_style))
        zone_data = [['Zone', 'City', 'Risk Score', 'Status']]
        for zone in report_data['zone_details'][:10]:
            zone_data.append([
                zone['zone'],
                zone['city'],
                f"{zone['risk_score']}%",
                zone['status']
            ])
        
        zone_table = Table(zone_data, colWidths=[2*inch, 2*inch, 1.5*inch, 1.5*inch])
        zone_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
        ]))
        story.append(zone_table)
        story.append(Spacer(1, 20))
    
    # Recommendations
    if report_data.get('recommendations'):
        story.append(Paragraph("Recommendations", heading_style))
        for i, rec in enumerate(report_data['recommendations'], 1):
            story.append(Paragraph(f"{i}. {rec}", normal_style))
        story.append(Spacer(1, 20))
    
    # Action Items
    if report_data.get('action_items'):
        story.append(Paragraph("Priority Action Items", heading_style))
        action_data = [['Priority', 'Action', 'Deadline']]
        for item in report_data['action_items']:
            action_data.append([
                item['priority'],
                item['action'],
                item['deadline']
            ])
        
        action_table = Table(action_data, colWidths=[1.5*inch, 3.5*inch, 2*inch])
        action_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        story.append(action_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route("/api/download-pdf-report")
def download_pdf_report():
    """Generate and download PDF report using real data from CSV"""
    report_type = request.args.get("type", "summary")
    city = request.args.get("city", "")
    start_date = request.args.get("start_date", "2025-07-01")
    end_date = request.args.get("end_date", "2025-12-31")
    
    if df is None or df.empty:
        return jsonify({"error": "No data available"}), 400
    
    # Filter data exactly like the UI does
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df["city"] == city]
    
    # Calculate real statistics from the actual data
    total_zones = len(filtered_df['zone'].unique()) if not filtered_df.empty else 0
    cities_covered = len(filtered_df['city'].unique()) if not filtered_df.empty else 0
    avg_risk_score = round(filtered_df['risk_score'].mean(), 1) if not filtered_df.empty else 0
    
    # Define risk categories based on actual data
    high_risk_zones = len(filtered_df[filtered_df['risk_score'] >= 70]) if not filtered_df.empty else 0
    medium_risk_zones = len(filtered_df[(filtered_df['risk_score'] >= 30) & (filtered_df['risk_score'] < 70)]) if not filtered_df.empty else 0
    low_risk_zones = len(filtered_df[filtered_df['risk_score'] < 30]) if not filtered_df.empty else 0
    
    # Determine overall risk level based on average
    overall_risk_level = "HIGH" if avg_risk_score >= 70 else "MEDIUM" if avg_risk_score >= 30 else "LOW"
    
    # Generate comprehensive report data using REAL data
    report_data = {
        "report_metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_type": report_type,
            "period": f"{start_date} to {end_date}",
            "total_zones_analyzed": total_zones,
            "cities_covered": cities_covered
        },
        "executive_summary": {
            "overall_risk_level": overall_risk_level,
            "high_risk_zones": high_risk_zones,
            "medium_risk_zones": medium_risk_zones,
            "low_risk_zones": low_risk_zones,
            "average_risk_score": avg_risk_score
        },
        "city_breakdown": [],
        "zone_details": [],
        "recommendations": [
            f"Focus immediate attention on {high_risk_zones} high-risk zones (≥70% risk score)",
            f"Enhance monitoring in {medium_risk_zones} medium-risk zones (30-69% risk score)",
            f"Maintain routine surveillance in {low_risk_zones} low-risk zones (<30% risk score)",
            f"Current state-wide average risk is {avg_risk_score}% - requires {overall_risk_level.lower()} priority response",
            "Strengthen inter-departmental coordination for rapid response"
        ],
        "action_items": [
            {"priority": "HIGH", "action": f"Deploy medical teams to {high_risk_zones} high-risk zones immediately", "deadline": "Within 24 hours"},
            {"priority": "MEDIUM", "action": f"Increase surveillance in {medium_risk_zones} medium-risk areas", "deadline": "Within 48 hours"},
            {"priority": "LOW", "action": "Review and update response protocols based on current data", "deadline": "Within 1 week"}
        ]
    }
    
    # City-wise breakdown using REAL data
    if not filtered_df.empty:
        city_stats = filtered_df.groupby('city').agg({
            'risk_score': ['mean', 'max', 'min', 'count'],
            'zone': 'nunique'
        }).round(2)
        
        for city_name in city_stats.index:
            avg_risk = city_stats.loc[city_name, ('risk_score', 'mean')]
            risk_level = "HIGH" if avg_risk >= 70 else "MEDIUM" if avg_risk >= 30 else "LOW"
            
            report_data["city_breakdown"].append({
                "city": city_name,
                "total_zones": int(city_stats.loc[city_name, ('zone', 'nunique')]),
                "avg_risk": round(avg_risk, 1),
                "max_risk": round(city_stats.loc[city_name, ('risk_score', 'max')], 1),
                "min_risk": round(city_stats.loc[city_name, ('risk_score', 'min')], 1),
                "risk_level": risk_level
            })
        
        # Zone details (sorted by risk score) - Top 20 highest risk zones
        zone_details = filtered_df.nlargest(20, 'risk_score')
        for _, row in zone_details.iterrows():
            risk_score = row['risk_score']
            status = "HIGH" if risk_score >= 70 else "MEDIUM" if risk_score >= 30 else "LOW"
            
            report_data["zone_details"].append({
                "zone": row['zone'],
                "city": row['city'],
                "risk_score": round(risk_score, 1),
                "status": status,
                "period": f"Month {int(row['month'])}, Week {int(row['week'])}"
            })
    
    try:
        # Generate PDF
        pdf_buffer = generate_pdf_report(report_data)
        
        # Create response
        response = make_response(pdf_buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=dengue_risk_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return response
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500

# -----------------------------
# Hospital Alert Interface
# -----------------------------
@app.route("/alerts")
def hospital_alerts():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hospital Alert System - Dengue Risk Intelligence</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    body { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f1f5f9; min-height: 100vh;
    }
    .navbar { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      border-bottom: 2px solid rgba(59, 130, 246, 0.3);
      backdrop-filter: blur(10px);
    }
    .brand { 
      font-weight: 700; color: #f1f5f9; font-size: 1.5rem;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .card { 
      background: rgba(15, 23, 42, 0.8); 
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 16px; backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .form-label { 
      color: #cbd5e1; font-weight: 600; font-size: 0.9rem;
      text-transform: uppercase; letter-spacing: 0.5px;
    }
    .form-select, .form-control, .form-check-input {
      background: rgba(30, 41, 59, 0.8); color: #f1f5f9 !important; 
      border: 2px solid rgba(71, 85, 105, 0.5);
      border-radius: 12px; transition: all 0.3s ease;
    }
    .form-select:focus, .form-control:focus { 
      box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25); 
      border-color: #3b82f6; background: rgba(30, 41, 59, 0.9);
      color: #f1f5f9 !important;
    }
    textarea.form-control {
      color: #f1f5f9 !important;
    }
    textarea.form-control::placeholder {
      color: #94a3b8 !important;
    }
    .btn-primary { 
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none; font-weight: 600; padding: 12px 24px;
      border-radius: 12px; transition: all 0.3s ease;
    }
    .btn-danger { 
      background: linear-gradient(135deg, #ef4444, #dc2626);
      border: none; font-weight: 600; padding: 12px 24px;
      border-radius: 12px; transition: all 0.3s ease;
    }
    .btn-warning { 
      background: linear-gradient(135deg, #f59e0b, #d97706);
      border: none; font-weight: 600; padding: 12px 24px;
      border-radius: 12px; transition: all 0.3s ease;
    }
    .hospital-card {
      background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 12px; padding: 16px; margin-bottom: 12px;
      transition: all 0.3s ease;
    }
    .hospital-card:hover {
      border-color: rgba(59, 130, 246, 0.4);
      background: rgba(30, 41, 59, 0.8);
    }
    .hospital-card.selected {
      border-color: #3b82f6; background: rgba(59, 130, 246, 0.1);
    }
    .section-title {
      font-size: 1.5rem; font-weight: 600; color: #f1f5f9;
      margin-bottom: 20px; display: flex; align-items: center;
    }
    .section-title i { margin-right: 12px; color: #3b82f6; }
    .alert-success {
      background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4);
      color: #10b981; border-radius: 12px;
    }
    .alert-info {
      background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.4);
      color: #3b82f6; border-radius: 12px;
    }
    .risk-badge {
      padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
      font-weight: 600; text-transform: uppercase;
    }
    .risk-low { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .risk-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .risk-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg mb-4">
    <div class="container-fluid">
      <span class="navbar-brand brand"><i class="fas fa-hospital me-2"></i>Hospital Alert System</span>
      <div class="ms-auto d-flex align-items-center">
        <a href="/dashboard" class="btn btn-outline-primary me-3">
          <i class="fas fa-chart-line me-2"></i>Dashboard
        </a>
        <a href="/" class="btn btn-outline-light">
          <i class="fas fa-home me-2"></i>Home
        </a>
      </div>
    </div>
  </nav>

  <div class="container-fluid px-4">
    <div class="row g-4">
      <!-- Alert Configuration -->
      <div class="col-12 col-lg-4">
        <div class="card p-4">
          <div class="section-title">
            <i class="fas fa-cog"></i>Alert Configuration
          </div>
          
          <div class="mb-3">
            <label class="form-label">City</label>
            <select id="alertCity" class="form-select">
              <option value="">Select City</option>
            </select>
          </div>

          <div class="mb-3">
            <label class="form-label">Zone</label>
            <select id="alertZone" class="form-select">
              <option value="">Select Zone</option>
            </select>
          </div>

          <div class="mb-3">
            <label class="form-label">Alert Type</label>
            <select id="alertType" class="form-select">
              <option value="low">Low Risk Alert</option>
              <option value="medium" selected>Medium Risk Alert</option>
              <option value="high">High Risk Alert</option>
              <option value="critical">Critical Alert</option>
            </select>
          </div>

          <div class="mb-3">
            <label class="form-label">Risk Score</label>
            <input type="number" id="riskScore" class="form-control" min="0" max="100" value="45" placeholder="Enter risk score">
          </div>

          <div class="mb-4">
            <label class="form-label">Custom Message</label>
            <textarea id="customMessage" class="form-control" rows="4" placeholder="Enter additional message for hospitals...">Dengue risk levels have increased in your area. Please ensure adequate preparation and resources for potential cases.</textarea>
          </div>

          <div class="d-grid gap-2">
            <button id="loadHospitals" class="btn btn-primary">
              <i class="fas fa-search me-2"></i>Load Hospitals
            </button>
            <button id="sendAlert" class="btn btn-danger" disabled>
              <i class="fas fa-bell me-2"></i>Send Alert
            </button>
          </div>
        </div>
      </div>

      <!-- Hospital Selection -->
      <div class="col-12 col-lg-8">
        <div class="card p-4">
          <div class="section-title">
            <i class="fas fa-hospital-alt"></i>Hospital Selection
          </div>
          
          <div id="alertStatus" style="display: none;"></div>
          
          <div class="mb-3">
            <button id="selectAll" class="btn btn-outline-primary btn-sm me-2">
              <i class="fas fa-check-double me-1"></i>Select All
            </button>
            <button id="deselectAll" class="btn btn-outline-secondary btn-sm">
              <i class="fas fa-times me-1"></i>Deselect All
            </button>
            <span id="selectedCount" class="ms-3 text-muted">0 hospitals selected</span>
          </div>

          <div id="hospitalsList">
            <div class="text-center text-muted py-5">
              <i class="fas fa-hospital fa-3x mb-3 opacity-50"></i>
              <p>Select city and zone, then click "Load Hospitals" to view available hospitals.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let hospitals = [];
    let selectedHospitals = [];

    // Load cities
    async function loadCities() {
      const res = await fetch('/api/cities');
      const cities = await res.json();
      const select = document.getElementById('alertCity');
      select.innerHTML = '<option value="">Select City</option>';
      cities.forEach(city => {
        select.innerHTML += `<option value="${city}">${city}</option>`;
      });
    }

    // Load zones
    async function loadZones() {
      const city = document.getElementById('alertCity').value;
      if (!city) return;
      
      const res = await fetch(`/api/zones?city=${encodeURIComponent(city)}`);
      const zones = await res.json();
      const select = document.getElementById('alertZone');
      select.innerHTML = '<option value="">All Zones</option>';
      zones.forEach(zone => {
        select.innerHTML += `<option value="${zone}">${zone}</option>`;
      });
    }

    // Load hospitals
    async function loadHospitals() {
      const city = document.getElementById('alertCity').value;
      const zone = document.getElementById('alertZone').value;
      
      if (!city) {
        alert('Please select a city first');
        return;
      }

      const res = await fetch(`/api/hospitals?city=${encodeURIComponent(city)}&zone=${encodeURIComponent(zone)}`);
      hospitals = await res.json();
      
      displayHospitals();
      document.getElementById('sendAlert').disabled = hospitals.length === 0;
    }

    // Display hospitals
    function displayHospitals() {
      const container = document.getElementById('hospitalsList');
      
      if (hospitals.length === 0) {
        container.innerHTML = `
          <div class="text-center text-muted py-5">
            <i class="fas fa-exclamation-triangle fa-3x mb-3 opacity-50"></i>
            <p>No hospitals found for the selected criteria.</p>
          </div>
        `;
        return;
      }

      container.innerHTML = hospitals.map(hospital => `
        <div class="hospital-card" data-id="${hospital.id}">
          <div class="form-check">
            <input class="form-check-input hospital-checkbox" type="checkbox" value="${hospital.id}" id="hospital${hospital.id}">
            <label class="form-check-label w-100" for="hospital${hospital.id}">
              <div class="d-flex justify-content-between align-items-start">
                <div>
                  <h6 class="mb-1 text-white">${hospital.name}</h6>
                  <p class="mb-1 text-muted small">
                    <i class="fas fa-map-marker-alt me-1"></i>${hospital.zone}, ${hospital.city}
                  </p>
                  <p class="mb-0 text-muted small">
                    <i class="fas fa-phone me-1"></i>${hospital.contact} | 
                    <i class="fas fa-envelope me-1"></i>${hospital.email}
                  </p>
                </div>
                <span class="risk-badge risk-medium">Ready</span>
              </div>
            </label>
          </div>
        </div>
      `).join('');

      // Add event listeners
      document.querySelectorAll('.hospital-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateSelection);
      });
    }

    // Update selection
    function updateSelection() {
      selectedHospitals = Array.from(document.querySelectorAll('.hospital-checkbox:checked')).map(cb => parseInt(cb.value));
      document.getElementById('selectedCount').textContent = `${selectedHospitals.length} hospitals selected`;
      
      // Update card styling
      document.querySelectorAll('.hospital-card').forEach(card => {
        const id = parseInt(card.dataset.id);
        if (selectedHospitals.includes(id)) {
          card.classList.add('selected');
        } else {
          card.classList.remove('selected');
        }
      });
    }

    // Send alert
    async function sendAlert() {
      if (selectedHospitals.length === 0) {
        alert('Please select at least one hospital');
        return;
      }

      const alertData = {
        alert_type: document.getElementById('alertType').value,
        city: document.getElementById('alertCity').value,
        zone: document.getElementById('alertZone').value,
        risk_score: parseInt(document.getElementById('riskScore').value),
        hospitals: selectedHospitals,
        message: document.getElementById('customMessage').value
      };

      try {
        const res = await fetch('/api/send-alert', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(alertData)
        });
        
        const result = await res.json();
        
        if (result.status === 'success') {
          document.getElementById('alertStatus').innerHTML = `
            <div class="alert alert-success">
              <i class="fas fa-check-circle me-2"></i>
              <strong>Alert Sent Successfully!</strong><br>
              ${result.message} at ${result.alert_details.timestamp}
            </div>
          `;
          document.getElementById('alertStatus').style.display = 'block';
          
          // Reset form
          setTimeout(() => {
            document.getElementById('alertStatus').style.display = 'none';
          }, 5000);
        }
      } catch (error) {
        document.getElementById('alertStatus').innerHTML = `
          <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Error:</strong> Failed to send alert. Please try again.
          </div>
        `;
        document.getElementById('alertStatus').style.display = 'block';
      }
    }

    // Event listeners
    document.getElementById('alertCity').addEventListener('change', loadZones);
    document.getElementById('loadHospitals').addEventListener('click', loadHospitals);
    document.getElementById('sendAlert').addEventListener('click', sendAlert);
    
    document.getElementById('selectAll').addEventListener('click', () => {
      document.querySelectorAll('.hospital-checkbox').forEach(cb => cb.checked = true);
      updateSelection();
    });
    
    document.getElementById('deselectAll').addEventListener('click', () => {
      document.querySelectorAll('.hospital-checkbox').forEach(cb => cb.checked = false);
      updateSelection();
    });

    // Initialize
    loadCities();
  </script>
</body>
</html>
"""

# -----------------------------
# Government Reports Interface
# -----------------------------
@app.route("/reports")
def government_reports():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Government Reports - Dengue Risk Intelligence</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    body { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f1f5f9; min-height: 100vh;
    }
    .navbar { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      border-bottom: 2px solid rgba(59, 130, 246, 0.3);
      backdrop-filter: blur(10px);
    }
    .brand { 
      font-weight: 700; color: #f1f5f9; font-size: 1.5rem;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .card { 
      background: rgba(15, 23, 42, 0.8); 
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 16px; backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .form-label { 
      color: #cbd5e1; font-weight: 600; font-size: 0.9rem;
      text-transform: uppercase; letter-spacing: 0.5px;
    }
    .form-select, .form-control {
      background: rgba(30, 41, 59, 0.8); color: #f1f5f9 !important; 
      border: 2px solid rgba(71, 85, 105, 0.5);
      border-radius: 12px; transition: all 0.3s ease;
    }
    .form-select:focus, .form-control:focus { 
      box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25); 
      border-color: #3b82f6; background: rgba(30, 41, 59, 0.9);
      color: #f1f5f9 !important;
    }
    textarea.form-control {
      color: #f1f5f9 !important;
    }
    textarea.form-control::placeholder {
      color: #94a3b8 !important;
    }
    .btn-primary { 
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none; font-weight: 600; padding: 12px 24px;
      border-radius: 12px; transition: all 0.3s ease;
    }
    .btn-success { 
      background: linear-gradient(135deg, #10b981, #059669);
      border: none; font-weight: 600; padding: 12px 24px;
      border-radius: 12px; transition: all 0.3s ease;
    }
    .section-title {
      font-size: 1.5rem; font-weight: 600; color: #f1f5f9;
      margin-bottom: 20px; display: flex; align-items: center;
    }
    .section-title i { margin-right: 12px; color: #3b82f6; }
    .report-card {
      background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 12px; padding: 20px; margin-bottom: 16px;
      transition: all 0.3s ease;
    }
    .report-card:hover {
      border-color: rgba(59, 130, 246, 0.4);
      background: rgba(30, 41, 59, 0.8);
    }
    .status-badge {
      padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
      font-weight: 600; text-transform: uppercase;
    }
    .status-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    .status-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .status-low { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .alert-success {
      background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4);
      color: #10b981; border-radius: 12px;
    }
    .email-tag {
      background: rgba(59, 130, 246, 0.2); color: #3b82f6;
      padding: 4px 8px; border-radius: 6px; font-size: 0.8rem;
      margin: 2px; display: inline-block;
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg mb-4">
    <div class="container-fluid">
      <span class="navbar-brand brand"><i class="fas fa-file-alt me-2"></i>Government Reports</span>
      <div class="ms-auto d-flex align-items-center">
        <a href="/alerts" class="btn btn-outline-warning me-3">
          <i class="fas fa-hospital me-2"></i>Hospital Alerts
        </a>
        <a href="/dashboard" class="btn btn-outline-primary me-3">
          <i class="fas fa-chart-line me-2"></i>Dashboard
        </a>
        <a href="/" class="btn btn-outline-light">
          <i class="fas fa-home me-2"></i>Home
        </a>
      </div>
    </div>
  </nav>

  <div class="container-fluid px-4">
    <div class="row g-4">
      <!-- Report Configuration -->
      <div class="col-12 col-lg-4">
        <div class="card p-4">
          <div class="section-title">
            <i class="fas fa-cog"></i>Report Configuration
          </div>
          
          <div class="mb-3">
            <label class="form-label">Report Type</label>
            <select id="reportType" class="form-select">
              <option value="summary">Executive Summary</option>
              <option value="detailed">Detailed Analysis</option>
              <option value="zone">Zone-wise Report</option>
              <option value="predictive">Predictive Report</option>
            </select>
          </div>

          <div class="mb-3">
            <label class="form-label">City Filter</label>
            <select id="reportCity" class="form-select">
              <option value="">All Cities</option>
            </select>
          </div>

          <div class="mb-3">
            <label class="form-label">Start Date</label>
            <input type="date" id="startDate" class="form-control" value="2025-07-01">
          </div>

          <div class="mb-3">
            <label class="form-label">End Date</label>
            <input type="date" id="endDate" class="form-control" value="2025-12-31">
          </div>

          <div class="d-grid gap-2 mb-4">
            <button id="generateReport" class="btn btn-primary">
              <i class="fas fa-chart-bar me-2"></i>Generate Report
            </button>
            <button id="downloadPdf" class="btn btn-outline-light" disabled>
              <i class="fas fa-file-pdf me-2"></i>Download PDF
            </button>
          </div>

          <hr class="text-secondary">

          <div class="section-title">
            <i class="fas fa-envelope"></i>Email Distribution
          </div>

          <div class="mb-3">
            <label class="form-label">Government Officials</label>
            <div id="emailTags" class="mb-2"></div>
            <input type="email" id="emailInput" class="form-control" placeholder="Enter email address">
            <button id="addEmail" class="btn btn-outline-primary btn-sm mt-2">
              <i class="fas fa-plus me-1"></i>Add Email
            </button>
          </div>

          <div class="d-grid">
            <button id="emailReport" class="btn btn-success" disabled>
              <i class="fas fa-paper-plane me-2"></i>Email Report
            </button>
          </div>
        </div>
      </div>

      <!-- Report Display -->
      <div class="col-12 col-lg-8">
        <div class="card p-4">
          <div class="section-title">
            <i class="fas fa-file-contract"></i>Generated Report
          </div>
          
          <div id="reportStatus" style="display: none;"></div>
          
          <div id="reportContent">
            <div class="text-center text-muted py-5">
              <i class="fas fa-file-alt fa-3x mb-3 opacity-50"></i>
              <p>Configure report settings and click "Generate Report" to create a comprehensive analysis.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let reportData = null;
    let emailList = [
      'health.minister@tn.gov.in',
      'chief.secretary@tn.gov.in',
      'director.health@tn.gov.in'
    ];

    // Initialize email tags
    function updateEmailTags() {
      const container = document.getElementById('emailTags');
      container.innerHTML = emailList.map(email => 
        `<span class="email-tag">${email} <i class="fas fa-times ms-1" onclick="removeEmail('${email}')"></i></span>`
      ).join('');
      
      document.getElementById('emailReport').disabled = emailList.length === 0 || !reportData;
    }

    // Add email
    function addEmail() {
      const input = document.getElementById('emailInput');
      const email = input.value.trim();
      
      if (email && email.includes('@') && !emailList.includes(email)) {
        emailList.push(email);
        input.value = '';
        updateEmailTags();
      }
    }

    // Remove email
    function removeEmail(email) {
      emailList = emailList.filter(e => e !== email);
      updateEmailTags();
    }

    // Load cities
    async function loadCities() {
      const res = await fetch('/api/cities');
      const cities = await res.json();
      const select = document.getElementById('reportCity');
      cities.forEach(city => {
        select.innerHTML += `<option value="${city}">${city}</option>`;
      });
    }

    // Generate report
    async function generateReport() {
      const reportType = document.getElementById('reportType').value;
      const city = document.getElementById('reportCity').value;
      const startDate = document.getElementById('startDate').value;
      const endDate = document.getElementById('endDate').value;

      try {
        const res = await fetch(`/api/generate-report?type=${reportType}&city=${encodeURIComponent(city)}&start_date=${startDate}&end_date=${endDate}`);
        reportData = await res.json();
        
        displayReport(reportData);
        document.getElementById('emailReport').disabled = emailList.length === 0;
        document.getElementById('downloadPdf').disabled = false;
        
      } catch (error) {
        document.getElementById('reportContent').innerHTML = `
          <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            Error generating report. Please try again.
          </div>
        `;
      }
    }

    // Download PDF report
    async function downloadPdfReport() {
      const reportType = document.getElementById('reportType').value;
      const city = document.getElementById('reportCity').value;
      const startDate = document.getElementById('startDate').value;
      const endDate = document.getElementById('endDate').value;

      const url = `/api/download-pdf-report?type=${reportType}&city=${encodeURIComponent(city)}&start_date=${startDate}&end_date=${endDate}`;
      
      try {
        // Create a temporary link to trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = `dengue_risk_report_${new Date().toISOString().slice(0,10)}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Show success message
        document.getElementById('reportStatus').innerHTML = `
          <div class="alert alert-success">
            <i class="fas fa-check-circle me-2"></i>
            <strong>PDF Download Started!</strong><br>
            Your dengue risk report is being downloaded.
          </div>
        `;
        document.getElementById('reportStatus').style.display = 'block';
        
        setTimeout(() => {
          document.getElementById('reportStatus').style.display = 'none';
        }, 3000);
        
      } catch (error) {
        document.getElementById('reportStatus').innerHTML = `
          <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            Error downloading PDF. Please try again.
          </div>
        `;
        document.getElementById('reportStatus').style.display = 'block';
      }
    }

    // Display report
    function displayReport(data) {
      const container = document.getElementById('reportContent');
      
      container.innerHTML = `

        <div class="row g-4">
          <div class="col-md-6">
            <h5 class="text-light mb-3"><i class="fas fa-city me-2"></i>City Breakdown</h5>
            ${data.city_breakdown.map(city => `
              <div class="report-card">
                <div class="d-flex justify-content-between align-items-center">
                  <div>
                    <h6 class="text-white mb-1">${city.city}</h6>
                    <small class="text-muted">${city.total_zones} zones analyzed</small>
                  </div>
                  <div class="text-end">
                    <div class="status-badge status-${city.risk_level.toLowerCase()}">${city.risk_level}</div>
                    <div class="small text-muted mt-1">${city.avg_risk}% avg risk</div>
                  </div>
                </div>
              </div>
            `).join('')}
          </div>

          <div class="col-md-6">
            <h5 class="text-light mb-3"><i class="fas fa-exclamation-triangle me-2"></i>High Risk Zones</h5>
            ${data.zone_details.slice(0, 5).map(zone => `
              <div class="report-card">
                <div class="d-flex justify-content-between align-items-center">
                  <div>
                    <h6 class="text-white mb-1">${zone.zone}, ${zone.city}</h6>
                    <small class="text-muted">${zone.period}</small>
                  </div>
                  <div class="text-end">
                    <div class="status-badge status-${zone.status.toLowerCase()}">${zone.status}</div>
                    <div class="small text-muted mt-1">${zone.risk_score}% risk</div>
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>

        <div class="mt-4">
          <h5 class="text-light mb-3"><i class="fas fa-lightbulb me-2"></i>Recommendations</h5>
          <ul class="list-unstyled">
            ${data.recommendations.map(rec => `
              <li class="mb-2 text-white"><i class="fas fa-arrow-right text-primary me-2"></i>${rec}</li>
            `).join('')}
          </ul>
        </div>

        <div class="mt-4">
          <h5 class="text-light mb-3"><i class="fas fa-tasks me-2"></i>Priority Actions</h5>
          ${data.action_items.map(item => `
            <div class="report-card">
              <div class="d-flex justify-content-between align-items-center">
                <div>
                  <h6 class="text-white mb-1">${item.action}</h6>
                  <small class="text-muted">${item.deadline}</small>
                </div>
                <span class="status-badge status-${item.priority.toLowerCase()}">${item.priority}</span>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    }

    // Email report
    async function emailReport() {
      if (!reportData || emailList.length === 0) return;

      const emailData = {
        recipients: emailList,
        report_type: document.getElementById('reportType').value,
        city: document.getElementById('reportCity').value
      };

      try {
        const res = await fetch('/api/email-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(emailData)
        });
        
        const result = await res.json();
        
        document.getElementById('reportStatus').innerHTML = `
          <div class="alert alert-success">
            <i class="fas fa-check-circle me-2"></i>
            <strong>Report Emailed Successfully!</strong><br>
            ${result.message} at ${result.details.sent_at}
          </div>
        `;
        document.getElementById('reportStatus').style.display = 'block';
        
        setTimeout(() => {
          document.getElementById('reportStatus').style.display = 'none';
        }, 5000);
        
      } catch (error) {
        document.getElementById('reportStatus').innerHTML = `
          <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            Error sending email. Please try again.
          </div>
        `;
        document.getElementById('reportStatus').style.display = 'block';
      }
    }

    // Event listeners
    document.getElementById('generateReport').addEventListener('click', generateReport);
    document.getElementById('downloadPdf').addEventListener('click', downloadPdfReport);
    document.getElementById('emailReport').addEventListener('click', emailReport);
    document.getElementById('addEmail').addEventListener('click', addEmail);
    document.getElementById('emailInput').addEventListener('keypress', (e) => {
      if (e.key === 'Enter') addEmail();
    });

    // Initialize
    loadCities();
    updateEmailTags();
  </script>
</body>
</html>
"""

# -----------------------------
# Landing Page
# -----------------------------
@app.route("/")
def landing():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dengue Risk Intelligence Platform</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    body { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
      color: #f1f5f9; min-height: 100vh; overflow-x: hidden;
    }
    .hero-section {
      min-height: 100vh; display: flex; align-items: center;
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
      position: relative; overflow: hidden;
    }
    .hero-section::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(59,130,246,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
      opacity: 0.3;
    }
    .hero-content { position: relative; z-index: 2; }
    .hero-title {
      font-size: 4rem; font-weight: 800; line-height: 1.1;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text; margin-bottom: 1.5rem;
    }
    .hero-subtitle {
      font-size: 1.5rem; font-weight: 400; color: #cbd5e1;
      margin-bottom: 2rem; line-height: 1.6;
    }
    .hero-description {
      font-size: 1.1rem; color: #94a3b8; margin-bottom: 3rem;
      line-height: 1.7; max-width: 600px;
    }
    .cta-button {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none; color: white; font-weight: 600; font-size: 1.1rem;
      padding: 16px 32px; border-radius: 12px; text-decoration: none;
      display: inline-flex; align-items: center; gap: 12px;
      transition: all 0.3s ease; box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }
    .cta-button:hover {
      background: linear-gradient(135deg, #1d4ed8, #1e40af);
      transform: translateY(-2px); color: white;
      box-shadow: 0 12px 48px rgba(59, 130, 246, 0.4);
    }
    .features-section {
      padding: 100px 0; background: rgba(15, 23, 42, 0.5);
      backdrop-filter: blur(10px);
    }
    .feature-card {
      background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 20px; padding: 40px 30px; text-align: center;
      transition: all 0.3s ease; backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2); height: 100%;
    }
    .feature-card:hover {
      border-color: rgba(59, 130, 246, 0.4); transform: translateY(-8px);
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .feature-icon {
      font-size: 3rem; margin-bottom: 24px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .feature-title {
      font-size: 1.5rem; font-weight: 700; color: #f1f5f9;
      margin-bottom: 16px;
    }
    .feature-description {
      color: #94a3b8; line-height: 1.6; font-size: 1rem;
    }
    .stats-section {
      padding: 80px 0; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
    }
    .stat-item {
      text-align: center; padding: 20px;
    }
    .stat-number {
      font-size: 3rem; font-weight: 800; color: #3b82f6;
      display: block; margin-bottom: 8px;
    }
    .stat-label {
      font-size: 1.1rem; color: #cbd5e1; font-weight: 500;
    }
    .navbar {
      background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(59, 130, 246, 0.2);
      position: fixed; top: 0; width: 100%; z-index: 1000;
    }
    .navbar-brand {
      font-weight: 700; font-size: 1.5rem;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .floating-elements {
      position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      pointer-events: none; overflow: hidden;
    }
    .floating-element {
      position: absolute; opacity: 0.1;
      animation: float 20s infinite linear;
    }
    @keyframes float {
      0% { transform: translateY(100vh) rotate(0deg); }
      100% { transform: translateY(-100px) rotate(360deg); }
    }
    .section-title {
      font-size: 2.5rem; font-weight: 700; text-align: center;
      margin-bottom: 60px; color: #f1f5f9;
    }
    .footer {
      background: rgba(15, 23, 42, 0.9); padding: 40px 0;
      border-top: 1px solid rgba(59, 130, 246, 0.2);
    }
  </style>
</head>
<body>
  <!-- Navigation -->
  <nav class="navbar navbar-expand-lg">
    <div class="container">
      <a class="navbar-brand" href="/">
        <i class="fas fa-shield-virus me-2"></i>Dengue Risk Intelligence
      </a>
      <div class="ms-auto">
        <a href="/reports" class="btn btn-outline-success me-3">
          <i class="fas fa-file-alt me-2"></i>Reports
        </a>
        <a href="/alerts" class="btn btn-outline-warning me-3">
          <i class="fas fa-hospital me-2"></i>Hospital Alerts
        </a>
        <a href="/dashboard" class="btn btn-outline-primary">
          <i class="fas fa-chart-line me-2"></i>Dashboard
        </a>
      </div>
    </div>
  </nav>

  <!-- Hero Section -->
  <section class="hero-section">
    <div class="floating-elements">
      <i class="fas fa-virus floating-element" style="top: 20%; left: 10%; font-size: 2rem; animation-delay: 0s;"></i>
      <i class="fas fa-map-marker-alt floating-element" style="top: 60%; left: 80%; font-size: 1.5rem; animation-delay: 5s;"></i>
      <i class="fas fa-chart-line floating-element" style="top: 40%; left: 15%; font-size: 1.8rem; animation-delay: 10s;"></i>
      <i class="fas fa-shield-alt floating-element" style="top: 70%; left: 70%; font-size: 2.2rem; animation-delay: 15s;"></i>
    </div>
    
    <div class="container">
      <div class="row align-items-center">
        <div class="col-lg-6">
          <div class="hero-content">
            <h1 class="hero-title">Dengue Risk Intelligence Platform</h1>
            <p class="hero-subtitle">Advanced AI-Powered Disease Surveillance & Risk Assessment</p>
            <p class="hero-description">
              Harness the power of machine learning and geospatial analytics to predict, monitor, and prevent dengue outbreaks. 
              Our platform provides real-time risk assessment, interactive mapping, and actionable insights for public health decision-making.
            </p>
            <a href="/dashboard" class="cta-button">
              <i class="fas fa-rocket"></i>
              Get Started
            </a>
          </div>
        </div>
        <div class="col-lg-6">
          <div class="text-center">
            <i class="fas fa-globe-americas" style="font-size: 20rem; color: rgba(59, 130, 246, 0.2);"></i>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Stats Section -->
  <section class="stats-section">
    <div class="container">
      <div class="row">
        <div class="col-md-3">
          <div class="stat-item">
            <span class="stat-number">95%</span>
            <span class="stat-label">Prediction Accuracy</span>
          </div>
        </div>
        <div class="col-md-3">
          <div class="stat-item">
            <span class="stat-number">5+</span>
            <span class="stat-label">Cities Monitored</span>
          </div>
        </div>
        <div class="col-md-3">
          <div class="stat-item">
            <span class="stat-number">24/7</span>
            <span class="stat-label">Real-time Monitoring</span>
          </div>
        </div>
        <div class="col-md-3">
          <div class="stat-item">
            <span class="stat-number">10+</span>
            <span class="stat-label">Zones Analyzed</span>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Features Section -->
  <section class="features-section">
    <div class="container">
      <h2 class="section-title">Powerful Features</h2>
      <div class="row g-4">
        <div class="col-lg-4 col-md-6">
          <div class="feature-card">
            <div class="feature-icon">
              <i class="fas fa-brain"></i>
            </div>
            <h3 class="feature-title">AI-Powered Predictions</h3>
            <p class="feature-description">
              Advanced machine learning algorithms analyze multiple data sources to predict dengue risk with 95% accuracy.
            </p>
          </div>
        </div>
        <div class="col-lg-4 col-md-6">
          <div class="feature-card">
            <div class="feature-icon">
              <i class="fas fa-map"></i>
            </div>
            <h3 class="feature-title">Interactive Risk Maps</h3>
            <p class="feature-description">
              Visualize risk levels across different zones with interactive maps, heatmaps, and real-time data overlays.
            </p>
          </div>
        </div>
        <div class="col-lg-4 col-md-6">
          <div class="feature-card">
            <div class="feature-icon">
              <i class="fas fa-chart-line"></i>
            </div>
            <h3 class="feature-title">Trend Analysis</h3>
            <p class="feature-description">
              Track risk patterns over time with comprehensive trend analysis and seasonal forecasting capabilities.
            </p>
          </div>
        </div>
        <div class="col-lg-4 col-md-6">
          <div class="feature-card">
            <div class="feature-icon">
              <i class="fas fa-hospital-alt"></i>
            </div>
            <h3 class="feature-title">Hospital Alert System</h3>
            <p class="feature-description">
              Municipality heads can instantly alert hospitals about dengue risk levels, ensuring rapid response and resource preparation.
            </p>
          </div>
        </div>
        <div class="col-lg-4 col-md-6">
          <div class="feature-card">
            <div class="feature-icon">
              <i class="fas fa-file-contract"></i>
            </div>
            <h3 class="feature-title">Government Reports</h3>
            <p class="feature-description">
              Generate comprehensive reports for Tamil Nadu government with zone-level risk analysis and email distribution to officials.
            </p>
          </div>
        </div>
        <div class="col-lg-4 col-md-6">
          <div class="feature-card">
            <div class="feature-icon">
              <i class="fas fa-database"></i>
            </div>
            <h3 class="feature-title">Comprehensive Data</h3>
            <p class="feature-description">
              Integrate weather, demographic, and epidemiological data for holistic risk assessment and analysis.
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="footer">
    <div class="container">
      <div class="row">
        <div class="col-md-6">
          <h5 class="text-primary">Dengue Risk Intelligence Platform</h5>
          <p class="text-white">Empowering public health through AI-driven disease surveillance and risk assessment.</p>
        </div>
        <div class="col-md-6 text-md-end">
          <p class="text-white mb-0">&copy; 2025 Dengue Risk Intelligence. All rights reserved.</p>
          <p class="text-white small">Powered by Advanced Machine Learning & Geospatial Analytics</p>
        </div>
      </div>
    </div>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# -----------------------------
# UI: Dashboard page
# -----------------------------
@app.route("/dashboard")
def home():
    if df is None or df.empty:
        return (
            "<link rel='stylesheet' "
            "href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css'/>"
            "<div class='container py-5'>"
            "<h2 class='mb-3'>Dengue Risk Intelligence</h2>"
            "<p>No prediction dataset found.</p>"
            "<p>Generate it by running: <code>python final_xg.py</code></p>"
            "</div>"
        )

    # Professional UI
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dengue Risk Intelligence Platform</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    body { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #f1f5f9; min-height: 100vh; overflow-x: hidden;
    }
    .navbar { 
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      border-bottom: 2px solid rgba(59, 130, 246, 0.3);
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .brand { 
      font-weight: 700; color: #f1f5f9; font-size: 1.5rem;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .card { 
      background: rgba(15, 23, 42, 0.8); 
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 16px; backdrop-filter: blur(10px);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
      transition: all 0.3s ease;
    }
    .card:hover { 
      border-color: rgba(59, 130, 246, 0.4);
      box-shadow: 0 12px 48px rgba(0,0,0,0.3);
      transform: translateY(-2px);
    }
    .form-label { 
      color: #cbd5e1; font-weight: 600; font-size: 0.9rem;
      text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
    }
    .form-select, .form-control {
      background: rgba(30, 41, 59, 0.8); color: #f1f5f9; 
      border: 2px solid rgba(71, 85, 105, 0.5);
      font-size: 0.95rem; min-height: 44px; border-radius: 12px;
      transition: all 0.3s ease; backdrop-filter: blur(5px);
    }
    .form-select:focus, .form-control:focus { 
      box-shadow: 0 0 0 0.25rem rgba(59, 130, 246, 0.25); 
      border-color: #3b82f6; background: rgba(30, 41, 59, 0.9);
    }
    .form-range { margin: 12px 0; }
    .form-range::-webkit-slider-thumb { 
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 50%; width: 20px; height: 20px;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .form-range::-moz-range-thumb { 
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 50%; width: 20px; height: 20px; border: none;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .btn-primary { 
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none; font-weight: 600; padding: 12px 24px;
      border-radius: 12px; transition: all 0.3s ease;
      box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    .btn-primary:hover { 
      background: linear-gradient(135deg, #1d4ed8, #1e40af);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }
    .btn-outline-light { 
      border: 2px solid rgba(71, 85, 105, 0.5); color: #cbd5e1; 
      font-weight: 600; padding: 12px 24px; border-radius: 12px;
      transition: all 0.3s ease; backdrop-filter: blur(5px);
    }
    .btn-outline-light:hover { 
      background: rgba(59, 130, 246, 0.1); color: #f1f5f9;
      border-color: #3b82f6; transform: translateY(-2px);
    }
    .kpi { 
      border-radius: 16px; padding: 20px; text-align: center;
      transition: all 0.3s ease; backdrop-filter: blur(10px);
      position: relative; overflow: hidden;
    }
    .kpi::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
      background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
      border-radius: 16px; z-index: 0;
    }
    .kpi > * { position: relative; z-index: 1; }
    .kpi:hover { transform: translateY(-4px); }
    .kpi-low { 
      background: linear-gradient(135deg, rgba(16,185,129,0.8), rgba(5,150,105,0.6));
      border: 3px solid rgba(16,185,129,0.9);
      box-shadow: 0 8px 24px rgba(16,185,129,0.4);
      color: #ffffff;
    }
    .kpi-med { 
      background: linear-gradient(135deg, rgba(245,158,11,0.8), rgba(217,119,6,0.6));
      border: 3px solid rgba(245,158,11,0.9);
      box-shadow: 0 8px 24px rgba(245,158,11,0.4);
      color: #ffffff;
    }
    .kpi-high { 
      background: linear-gradient(135deg, rgba(239,68,68,0.8), rgba(220,38,38,0.6));
      border: 3px solid rgba(239,68,68,0.9);
      box-shadow: 0 8px 24px rgba(239,68,68,0.4);
      color: #ffffff;
    }
    .legend-dot { 
      display: inline-block; width: 14px; height: 14px; 
      border-radius: 50%; margin-right: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .legend { 
      font-size: 0.95rem; color: #e2e8f0; line-height: 2.2;
      background: rgba(30, 41, 59, 0.6); padding: 20px;
      border-radius: 12px; backdrop-filter: blur(5px);
      border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .legend-item {
      display: flex; align-items: center; margin-bottom: 12px;
      padding: 12px 16px; border-radius: 8px;
      background: rgba(15, 23, 42, 0.6);
      transition: all 0.3s ease;
      border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .legend-item:hover {
      background: rgba(59, 130, 246, 0.15);
      transform: translateX(4px);
      border-color: rgba(59, 130, 246, 0.4);
    }
    .legend-item strong {
      color: #f1f5f9;
      font-size: 1rem;
      font-weight: 600;
    }
    .legend-item small {
      color: #cbd5e1;
      font-size: 0.85rem;
    }
    iframe { 
      width: 100%; height: 75vh; border: none; 
      border-radius: 16px; background: #0f172a;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .loading { 
      display: none; position: absolute; inset: 0; 
      background: rgba(0,0,0,0.6); align-items: center; 
      justify-content: center; border-radius: 16px;
      backdrop-filter: blur(5px);
    }
    .spinner-border { color: #3b82f6; }
    .section-title {
      font-size: 1.1rem; font-weight: 600; color: #f1f5f9;
      margin-bottom: 16px; display: flex; align-items: center;
    }
    .section-title i { margin-right: 8px; color: #3b82f6; }
    .month-display {
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text; font-weight: 600; font-size: 1rem;
    }
    .status-badge {
      position: absolute; top: 16px; right: 16px;
      background: rgba(59, 130, 246, 0.2); color: #3b82f6;
      padding: 4px 12px; border-radius: 20px; font-size: 0.8rem;
      font-weight: 500; border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .filter-section { margin-bottom: 24px; }
    .kpi-number { 
      font-size: 2.2rem; font-weight: 800; margin-bottom: 4px; 
      color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .kpi-label { 
      font-size: 0.9rem; font-weight: 600; opacity: 1;
      color: #ffffff; text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    .chart-container {
      position: relative; height: 300px; width: 100%;
      background: rgba(15, 23, 42, 0.3); border-radius: 12px;
      padding: 16px; backdrop-filter: blur(5px);
    }
    .chart-container canvas {
      border-radius: 8px;
    }
    .chart-container-large {
      position: relative; height: 400px; width: 100%;
      background: rgba(15, 23, 42, 0.3); border-radius: 12px;
      padding: 20px; backdrop-filter: blur(5px);
      border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .chart-container-large canvas {
      border-radius: 8px;
    }
    .chart-container-enhanced {
      position: relative; height: 450px; width: 100%;
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.4), rgba(30, 41, 59, 0.3));
      border-radius: 16px; padding: 24px; backdrop-filter: blur(10px);
      border: 2px solid rgba(59, 130, 246, 0.3);
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .chart-container-enhanced canvas {
      border-radius: 12px;
    }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg mb-4">
  <div class="container-fluid">
    <span class="navbar-brand brand"><i class="fas fa-shield-virus me-2"></i>Dengue Risk Intelligence</span>
    <div class="ms-auto d-flex align-items-center">
      <a href="/" class="btn btn-outline-light me-3">
        <i class="fas fa-home me-2"></i>Home
      </a>
      <span class="text-secondary ms-3">H2 2025 Predictions</span>
    </div>
  </div>
</nav>

<div class="container-fluid px-4">
  <div class="row g-4">
    <!-- Analysis Filters Column -->
    <div class="col-12 col-xl-3">
      <div class="card p-4">
        <div class="section-title">
          <i class="fas fa-filter"></i>Analysis Filters
        </div>
        
        <div class="filter-section">
          <label class="form-label"><i class="fas fa-city me-2"></i>City</label>
          <select id="city" class="form-select"></select>
        </div>

        <div class="filter-section">
          <label class="form-label"><i class="fas fa-map-marker-alt me-2"></i>Zone</label>
          <select id="zone" class="form-select"></select>
        </div>

        <div class="filter-section">
          <label class="form-label"><i class="fas fa-calendar-alt me-2"></i>Month (Jul–Dec)</label>
          <input type="range" min="7" max="12" step="1" value="7" class="form-range" id="monthRange">
          <div class="d-flex justify-content-between text-muted small">
            <span>Jul</span><span>Aug</span><span>Sep</span><span>Oct</span><span>Nov</span><span>Dec</span>
          </div>
          <div class="mt-2 text-center">
            <span id="monthDisplay" class="month-display">Month: July</span>
          </div>
        </div>

        <div class="filter-section">
          <label for="week" class="form-label"><i class="fas fa-clock me-2"></i>Week</label>
          <select id="week" class="form-select"></select>
        </div>

        <div class="filter-section">
          <label class="form-label"><i class="fas fa-layer-group me-2"></i>Visualization</label>
          <select id="layer" class="form-select">
            <option value="markers"><i class="fas fa-map-pin"></i> Risk Markers</option>
            <option value="heatmap"><i class="fas fa-fire"></i> Heat Map</option>
          </select>
        </div>

        <div class="d-grid gap-3 mt-4">
          <button id="showMap" class="btn btn-primary">
            <i class="fas fa-sync-alt me-2"></i>Update Analysis
          </button>
          <button id="downloadCsv" class="btn btn-outline-light">
            <i class="fas fa-download me-2"></i>Export Data
          </button>
        </div>
      </div>
    </div>

    <!-- Interactive Map Column (Right of Analysis Filters) -->
    <div class="col-12 col-xl-9">
      <div class="card p-3 position-relative">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <div class="section-title mb-0">
            <i class="fas fa-map"></i>Interactive Risk Map
          </div>
          <div class="text-muted small">
            <i class="fas fa-info-circle me-1"></i>Click markers for insights
          </div>
        </div>
        <div class="loading" id="loading">
          <div class="text-center">
            <div class="spinner-border mb-3" role="status"></div>
            <div class="text-light">Loading risk analysis...</div>
          </div>
        </div>
        <iframe id="mapFrame" title="Dengue Risk Intelligence Map"></iframe>
      </div>
    </div>
  </div>

  <!-- Risk Distribution Section (Horizontal) -->
  <div class="row g-4 mt-2">
    <div class="col-12">
      <div class="card p-4">
        <div class="section-title">
          <i class="fas fa-chart-pie"></i>Risk Distribution
        </div>
        <div class="row g-3">
          <div class="col-12 col-md-4">
            <div class="kpi kpi-low">
              <div class="kpi-number" id="kpiLow">0</div>
              <div class="kpi-label"><i class="fas fa-shield-alt me-1"></i>Low Risk Zones</div>
            </div>
          </div>
          <div class="col-12 col-md-4">
            <div class="kpi kpi-med">
              <div class="kpi-number" id="kpiMed">0</div>
              <div class="kpi-label"><i class="fas fa-exclamation-triangle me-1"></i>Medium Risk Zones</div>
            </div>
          </div>
          <div class="col-12 col-md-4">
            <div class="kpi kpi-high">
              <div class="kpi-number" id="kpiHigh">0</div>
              <div class="kpi-label"><i class="fas fa-exclamation-circle me-1"></i>High Risk Zones</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Risk Levels Guide Section -->
  <div class="row g-4 mt-2">
    <div class="col-12">
      <div class="card p-4">
        <div class="section-title">
          <i class="fas fa-info-circle"></i>Risk Levels Guide
        </div>
        <div class="row g-3">
          <div class="col-12 col-md-4">
            <div class="legend-item">
              <span class="legend-dot" style="background: linear-gradient(135deg, #10b981, #059669)"></span>
              <div>
                <strong>Low Risk</strong> (&lt; 30)<br>
                <small>Routine surveillance recommended</small>
              </div>
            </div>
          </div>
          <div class="col-12 col-md-4">
            <div class="legend-item">
              <span class="legend-dot" style="background: linear-gradient(135deg, #f59e0b, #d97706)"></span>
              <div>
                <strong>Medium Risk</strong> (30–69)<br>
                <small>Enhanced monitoring required</small>
              </div>
            </div>
          </div>
          <div class="col-12 col-md-4">
            <div class="legend-item">
              <span class="legend-dot" style="background: linear-gradient(135deg, #ef4444, #dc2626)"></span>
              <div>
                <strong>High Risk</strong> (≥ 70)<br>
                <small>Immediate intervention needed</small>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  </div>

  <!-- Enhanced Trend Chart Section (Full Width Below) -->
  <div class="row g-4 mt-2">
    <div class="col-12">
      <div class="card p-4">
        <div class="section-title">
          <i class="fas fa-chart-line"></i>Enhanced Risk Trend Analysis
        </div>
        <div class="chart-container-enhanced">
          <canvas id="trendChart"></canvas>
        </div>
      </div>
    </div>
  </div>
  </div>
</div>

<script>
const storage = window.localStorage;

function persist(key, value) {
  storage.setItem(key, value);
}
function readPersist(key, fallback=null) {
  return storage.getItem(key) ?? fallback;
}

async function loadCities() {
  const res = await fetch('/api/cities');
  const cities = await res.json();
  const cityEl = document.getElementById('city');
  cityEl.innerHTML = cities.map(c => `<option value="${c}">${c}</option>`).join('');
  const savedCity = readPersist('city', cities[0] || '');
  if (savedCity) cityEl.value = savedCity;
  await loadZones();
}

async function loadZones() {
  const city = document.getElementById('city').value;
  const res = await fetch('/api/zones?city=' + encodeURIComponent(city));
  const zones = await res.json();
  const zoneEl = document.getElementById('zone');
  zoneEl.innerHTML = zones.map(z => `<option value="${z}">${z}</option>`).join('');
  const savedZone = readPersist('zone', zones[0] || 'All');
  if (savedZone) zoneEl.value = savedZone;
  await loadWeeks();
}

async function loadWeeks() {
  const city = document.getElementById('city').value;
  const zone = document.getElementById('zone').value;
  const month = document.getElementById('monthRange').value;
  const res = await fetch(`/api/weeks?city=${encodeURIComponent(city)}&zone=${encodeURIComponent(zone)}&month=${encodeURIComponent(month)}`);
  const weeks = await res.json();
  const weekEl = document.getElementById('week');
  weekEl.innerHTML = weeks.map(w => `<option value="${w}">${w}</option>`).join('');
  const savedWeek = readPersist('week', weeks[0] || '');
  if (savedWeek && weeks.includes(parseInt(savedWeek))) weekEl.value = savedWeek;
}

async function updateSummary() {
  const city = document.getElementById('city').value;
  const zone = document.getElementById('zone').value;
  const month = document.getElementById('monthRange').value;
  const week = document.getElementById('week').value;
  const res = await fetch(`/api/summary?city=${encodeURIComponent(city)}&zone=${encodeURIComponent(zone)}&month=${encodeURIComponent(month)}&week=${encodeURIComponent(week)}`);
  const s = await res.json();
  document.getElementById('kpiLow').innerText = s.low;
  document.getElementById('kpiMed').innerText = s.medium;
  document.getElementById('kpiHigh').innerText = s.high;
}

let trendChart = null;

async function updateTrendChart() {
  const city = document.getElementById('city').value;
  const zone = document.getElementById('zone').value;
  const res = await fetch(`/api/trend?city=${encodeURIComponent(city)}&zone=${encodeURIComponent(zone)}`);
  const trendData = await res.json();
  
  if (trendChart) {
    trendChart.destroy();
  }
  
  const ctx = document.getElementById('trendChart').getContext('2d');
  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: trendData.map(d => d.period),
      datasets: [{
        label: 'Average Risk Score',
        data: trendData.map(d => d.avg_risk),
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 4,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: trendData.map(d => {
          if (d.avg_risk >= 70) return '#ef4444';
          if (d.avg_risk >= 30) return '#f59e0b';
          return '#10b981';
        }),
        pointBorderColor: '#ffffff',
        pointBorderWidth: 3,
        pointRadius: 8,
        pointHoverRadius: 12,
        pointHoverBorderWidth: 4
      }, {
        label: 'Risk Threshold - High (70%)',
        data: Array(trendData.length).fill(70),
        borderColor: 'rgba(239, 68, 68, 0.6)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false
      }, {
        label: 'Risk Threshold - Medium (30%)',
        data: Array(trendData.length).fill(30),
        borderColor: 'rgba(245, 158, 11, 0.6)',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#e2e8f0',
            font: { size: 12, weight: '500' }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          borderColor: 'rgba(59, 130, 246, 0.7)',
          borderWidth: 2,
          cornerRadius: 8,
          titleFont: { size: 14, weight: 'bold' },
          bodyFont: { size: 13 },
          padding: 12,
          callbacks: {
            title: function(context) {
              return `Period: ${context[0].label}`;
            },
            label: function(context) {
              if (context.datasetIndex === 0) {
                const risk = context.parsed.y;
                let riskLevel = 'Low';
                if (risk >= 70) riskLevel = 'High';
                else if (risk >= 30) riskLevel = 'Medium';
                return `Risk Score: ${risk.toFixed(1)}% (${riskLevel})`;
              }
              return context.dataset.label;
            },
            afterBody: function(context) {
              if (context[0].datasetIndex === 0) {
                const dataIndex = context[0].dataIndex;
                const zoneCount = trendData[dataIndex].zone_count;
                return [`Zones analyzed: ${zoneCount}`, `Click point for details`];
              }
              return [];
            }
          }
        }
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 11 } },
          grid: { color: 'rgba(71, 85, 105, 0.3)' }
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { 
            color: '#94a3b8',
            font: { size: 11 },
            callback: function(value) {
              return value + '%';
            }
          },
          grid: { color: 'rgba(71, 85, 105, 0.3)' }
        }
      },
      elements: {
        point: {
          hoverBorderWidth: 3
        }
      }
    }
  });
}

function updateMap() {
  const city = document.getElementById('city').value;
  const zone = document.getElementById('zone').value;
  const month = document.getElementById('monthRange').value;
  const week = document.getElementById('week').value;
  const layer = document.getElementById('layer').value;
  persist('city', city); persist('zone', zone); persist('month', month); persist('week', week); persist('layer', layer);
  const url = `/map?city=${encodeURIComponent(city)}&zone=${encodeURIComponent(zone)}&month=${encodeURIComponent(month)}&week=${encodeURIComponent(week)}&layer=${encodeURIComponent(layer)}&_ts=${Date.now()}`;
  const frame = document.getElementById('mapFrame');
  const loading = document.getElementById('loading');
  loading.style.display = 'flex';
  frame.onload = () => { loading.style.display = 'none'; };
  frame.src = url;
  updateSummary();
  updateTrendChart();
}

document.getElementById('city').addEventListener('change', async () => { await loadZones(); updateMap(); });
document.getElementById('zone').addEventListener('change', async () => { await loadWeeks(); updateMap(); });
document.getElementById('layer').addEventListener('change', updateMap);
document.getElementById('monthRange').addEventListener('input', (e) => {
  const months = ['', '', '', '', '', '', '', 'July', 'August', 'September', 'October', 'November', 'December'];
  document.getElementById('monthDisplay').innerText = 'Month: ' + months[parseInt(e.target.value)];
});
document.getElementById('monthRange').addEventListener('change', async () => { await loadWeeks(); updateMap(); });
document.getElementById('week').addEventListener('change', updateMap);
document.getElementById('showMap').addEventListener('click', updateMap);
document.getElementById('downloadCsv').addEventListener('click', () => {
  const city = document.getElementById('city').value;
  const zone = document.getElementById('zone').value;
  const month = document.getElementById('monthRange').value;
  const week = document.getElementById('week').value;
  window.open(`/api/download?city=${encodeURIComponent(city)}&zone=${encodeURIComponent(zone)}&month=${encodeURIComponent(month)}&week=${encodeURIComponent(week)}`, '_blank');
});

// Initialize
(async () => {
  await loadCities();
  const monthNames = ['', '', '', '', '', '', '', 'July', 'August', 'September', 'October', 'November', 'December'];
  const savedMonth = readPersist('month', '7');
  document.getElementById('monthRange').value = savedMonth;
  document.getElementById('monthDisplay').innerText = 'Month: ' + monthNames[parseInt(savedMonth)];
  const savedLayer = readPersist('layer', 'markers');
  document.getElementById('layer').value = savedLayer;
  await loadWeeks();
  const savedWeek = readPersist('week', document.getElementById('week').value);
  if (savedWeek) document.getElementById('week').value = savedWeek;
  updateMap();
  updateTrendChart();
})();
</script>
</body>
</html>
"""

# -----------------------------
# Map route with filters
# -----------------------------
@app.route("/map")
def map_filtered():
    if df is None or df.empty:
        return "<h3>No data available</h3>", 404
    city = request.args.get("city", type=str)
    zone = request.args.get("zone", type=str)  # "All" or specific
    month = request.args.get("month", type=int)
    week = request.args.get("week", type=int)
    layer = request.args.get("layer", type=str) or "markers"
    try:
        city_map_path = create_map(city, zone, month, week=week, layer=layer)
        with open(city_map_path, "r", encoding="utf-8") as f:
            html = f.read()
        from flask import make_response
        resp = make_response(html)
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>", 404

# -----------------------------
# Predict API (unchanged usage)
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    try:
        pred_df = predict_risk(input_df)
        risk = float(pred_df.iloc[0]["risk_score"])
        return jsonify({"zone": data.get("zone", "unknown"), "risk_score": risk})
    except Exception as e:
        return jsonify({ "error": str(e) }), 400

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
