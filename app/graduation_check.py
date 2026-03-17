import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from dateutil.parser import parse as parse_date
from collections import defaultdict
from google.genai import types
from . import global_state

GRADE_POINTS = {
    'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C+': 2.5,
    'C': 2.0, 'D+': 1.5, 'D': 1.0, 'F': 0.0
}
NO_CREDIT_GRADES = {'F', 'W', 'NP', 'U'} 
PENDING_GRADES = {'N', 'I'}
NON_EARNING_GRADES = NO_CREDIT_GRADES | PENDING_GRADES

GRADUATION_CHECK_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="check_graduation_status",
            description="Checks graduation requirements using pre-read transcript data, activity hours, payment status, and the specific curriculum rules provided.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "student_id": types.Schema(type=types.Type.STRING),
                    "student_name": types.Schema(type=types.Type.STRING),
                    "faculty": types.Schema(type=types.Type.STRING),
                    "field_of_study": types.Schema(type=types.Type.STRING),
                    "admission_year": types.Schema(type=types.Type.INTEGER),
                    "transcript_data": types.Schema(
                        type=types.Type.ARRAY, 
                        description="List of all courses taken. VERY IMPORTANT: Do NOT include course names to save processing time. Only extract code, grade, and credit.",
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "code": types.Schema(type=types.Type.STRING, description="Course code e.g. 01418111"),
                                "grade": types.Schema(type=types.Type.STRING, description="Grade received e.g. A, B+, P, N"),
                                "credit": types.Schema(type=types.Type.INTEGER, description="Course credits e.g. 3")
                            }
                        )
                    ),
                    "final_cumulative_gpa": types.Schema(type=types.Type.NUMBER),
                    "final_total_credits": types.Schema(type=types.Type.INTEGER),
                    "semester_gpas": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT)),
                    "activity_data": types.Schema(
                        type=types.Type.ARRAY,
                        description="A list of activities, each with a category and hours.",
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "category": types.Schema(type=types.Type.STRING),
                                "hours": types.Schema(type=types.Type.INTEGER)
                            }
                        )
                    ),
                    "payment_status_clear": types.Schema(type=types.Type.BOOLEAN),
                    "payment_amount": types.Schema(type=types.Type.NUMBER),
                    "payment_date": types.Schema(type=types.Type.STRING),
                    "payment_channel": types.Schema(type=types.Type.STRING),
                    "payment_term_year_semester": types.Schema(type=types.Type.STRING)
                },
                required=["student_id", "student_name", "transcript_data", "final_cumulative_gpa", "final_total_credits", "semester_gpas"]
            ),
        )
    ]
)

def _extract_credits(text_value: str, default: int = 0) -> int:
    if not isinstance(text_value, str): return default
    match = re.search(r'\d+', text_value)
    return int(match.group(0)) if match else default

def _normalize_term(term_str: str) -> str:
    term_str = term_str.lower()
    if 'first' in term_str or 'ภาคต้น' in term_str: return '1'
    if 'second' in term_str or 'ภาคปลาย' in term_str: return '2'
    if 'summer' in term_str or 'ฤดูร้อน' in term_str: return 'S'
    return '0'

def _calculate_stats(transcript_data: List[Dict]) -> Dict:
    total_points = 0.0
    total_credits_gpa = 0
    total_credits_earned = 0
    pending_courses = []

    for course in transcript_data:
        grade = course.get('grade', '').strip().upper()
        credit_raw = str(course.get('credit', course.get('credits', '0')))
        credit = _extract_credits(credit_raw, 0)
            
        # เช็ควิชาที่ยังไม่จบ
        if grade in PENDING_GRADES:
            pending_courses.append(f"{course.get('code')} ({course.get('name')})")
            continue

        # นับหน่วยกิตสะสม (ผ่าน)
        if grade not in NO_CREDIT_GRADES and grade not in PENDING_GRADES:
            total_credits_earned += credit

        # คำนวณ GPA (เฉพาะเกรดที่มีแต้ม)
        if grade in GRADE_POINTS:
            total_points += (GRADE_POINTS[grade] * credit)
            total_credits_gpa += credit

    calculated_gpa = (total_points / total_credits_gpa) if total_credits_gpa > 0 else 0.00
    
    return {
        "calculated_gpa": calculated_gpa,
        "calculated_total_credits": total_credits_earned,
        "pending_courses": pending_courses
    }
    
def _check_payment(payment_info: Dict, latest_transcript_term: str) -> Dict:
    """ตรวจสอบใบเสร็จเทียบกับเทอมล่าสุดในทรานสคริปต์"""
    if not payment_info.get("amount"):
        return {"status": "ไม่ผ่านเกณฑ์", "details": "ไม่พบยอดเงินในใบเสร็จ", "unmet": "ไม่พบหลักฐานการชำระเงิน"}

    receipt_term_str = payment_info.get("term_year_semester", "")

    is_match = False
    if latest_transcript_term and receipt_term_str:
        normalized_receipt = _normalize_term(receipt_term_str)
        normalized_transcript = _normalize_term(latest_transcript_term)
        

        if normalized_receipt == normalized_transcript and any(y in receipt_term_str for y in latest_transcript_term.split()):
             is_match = True

    details = f"ยอดชำระ {payment_info.get('amount')} บาท สำหรับ {receipt_term_str}"
    
    if not is_match:
        # *จุดสำคัญ* ถ้าเทอมไม่ตรง ให้แจ้งเตือนแบบตัวอย่างที่ผมทำ
        return {
            "status": "ไม่ผ่านเกณฑ์",
            "details": details,
            "unmet": f"ใบเสร็จเป็นของ {receipt_term_str} แต่เทอมล่าสุดในทรานสคริปต์คือ {latest_transcript_term} (ต้องใช้ใบเสร็จของเทอมล่าสุด)"
        }
        
    return {"status": "ผ่านเกณฑ์", "details": details, "unmet": None}

def _check_activities(activity_data: Optional[List[Dict]]) -> Dict:
    REQUIRED_ACTIVITY_HOURS = 17 
    if not activity_data:
        return { "status": "ไม่ผ่านเกณฑ์", 
                "details": "ไม่พบข้อมูลกิจกรรม", 
                "unmet_requirement": "ไม่พบข้อมูลกิจกรรม กรุณายื่นเอกสารกิจกรรมให้ครบถ้วน", 
                "total_hours": 0, 
                "breakdown": {} }
    
    total_hours = sum(act.get('hours', 0) for act in activity_data)
    # remaining_hours = max(0, REQUIRED_ACTIVITY_HOURS - total_hours)
    
    breakdown = defaultdict(int)
    for act in activity_data:
        category = act.get('category', 'ไม่ระบุ')
        if 'มหาวิทยาลัย' in category: 
            breakdown['กิจกรรมมหาวิทยาลัย'] += act.get('hours', 0)
        elif 'สมรรถนะ' in category: 
            breakdown['กิจกรรมเพื่อเสริมสร้างสมรรถนะ'] += act.get('hours', 0)
        elif 'สังคม' in category: 
            breakdown['กิจกรรมเพื่อสังคม'] += act.get('hours', 0)
        else: 
            breakdown[category] += act.get('hours', 0)
            
    details_summary = f"มีชั่วโมงกิจกรรมรวม {total_hours} ชั่วโมง"
    if total_hours < REQUIRED_ACTIVITY_HOURS:
        return { "status": "ไม่ผ่านเกณฑ์", 
                "details": details_summary, 
                "unmet_requirement": f"ชั่วโมงกิจกรรมรวมไม่ครบตามเกณฑ์ ({total_hours}/{REQUIRED_ACTIVITY_HOURS} ชั่วโมง)", 
                "total_hours": total_hours, 
                "breakdown": dict(breakdown) }
        
    return { "status": "ผ่านเกณฑ์", 
            "details": details_summary, 
            "unmet_requirement": None, 
            "total_hours": total_hours, 
            "breakdown": dict(breakdown) }

def check_graduation_status(
    student_id: str,
    student_name: str,
    transcript_data: List[Dict],
    final_cumulative_gpa: float,
    final_total_credits: int,
    semester_gpas: List[Dict],
    activity_data: Optional[List[Dict]] = None,
    payment_status_clear: Optional[bool] = None,
    payment_amount: Optional[float] = None,
    payment_date: Optional[str] = None,
    payment_channel: Optional[str] = None,
    payment_term_year_semester: Optional[str] = None
):
    if not transcript_data and not semester_gpas: 
        return { 
            "error": "missing_documents", 
            "transcript_found": False, 
            "activity_found": bool(activity_data), 
            "payment_found": bool(payment_amount) 
        }
        
    curriculum_rules = global_state.curriculum_rules_json
    unmet_requirements = []
    
    stats = _calculate_stats(transcript_data)
    real_gpa = stats['calculated_gpa']
    real_credits = stats['calculated_total_credits']
    pending_courses = stats['pending_courses']
    
    gpa_display = f"{real_gpa:.2f}"
    
    if not curriculum_rules:
        unmet_requirements.append("ไม่สามารถอ่านข้อมูลหลักสูตรได้")
        
    if pending_courses:
        for course in pending_courses:
            unmet_requirements.append(f"รายวิชายังไม่สมบูรณ์: {course} (เกรด N/I)")

    # 4. ตรวจสอบเกรดเฉลี่ย (GPA)
    if real_gpa < 2.00:
        unmet_requirements.append(f"GPA สะสม ({gpa_display}) ต่ำกว่าเกณฑ์ขั้นต่ำ 2.00")

    # 5. ตรวจสอบหน่วยกิตรวม
    # ดึงเกณฑ์ขั้นต่ำจากหลักสูตร (ถ้าหาไม่เจอให้ใช้ค่า Default 124)
    min_total_credits_rule = 124
    if curriculum_rules:
        try:
            txt = curriculum_rules.get('หลักสูตร', {}).get('จำนวนหน่วยกิต', '124')
            min_total_credits_rule = _extract_credits(txt, 124)
        except:
            pass
            
    if real_credits < min_total_credits_rule:
        unmet_requirements.append(f"หน่วยกิตรวม ({real_credits}) ต่ำกว่าเกณฑ์ขั้นต่ำ {min_total_credits_rule} หน่วยกิต")
        
    if unmet_requirements:
        return {
            "student_id": student_id, "student_name": student_name,
            "is_eligible_for_graduation": False,
            "overall_summary_message": "ไม่สามารถตรวจสอบการจบการศึกษาได้เนื่องจากข้อมูลไม่ครบถ้วน",
            "cumulative_gpa": gpa_display,
            "total_credits_earned": final_total_credits,
            "unmet_requirements": unmet_requirements,
            "semesters_summary": {}, "activity_summary": {}, "payment_summary": {}
        }
 
    # Course requirements
    course_records = {}
    for c in transcript_data:
        code = (c.get('course_code') or c.get('code'))
        if code and isinstance(code, str): # Ensure code is a string
            c['credits'] = c.get('credits') or c.get('credit')
            course_records[code.strip()] = c
    structure = curriculum_rules.get('หลักสูตร', {}).get('โครงสร้างหลักสูตร', {})
    
    # Required courses check
    required_courses_list = curriculum_rules.get('หลักสูตร', {}).get('รายวิชา', {}).get('หมวดวิชาเฉพาะ', {}).get('วิชาเฉพาะบังคับ', {}).get('กลุ่ม', [])
    all_required_codes = []
    for group in required_courses_list:
        for course in group.get('รายวิชา', []):
            all_required_codes.append(course)
    for course in curriculum_rules.get('หลักสูตร', {}).get('รายวิชา', {}).get('หมวดวิชาเฉพาะ', {}).get('วิชาแกน', []):
        all_required_codes.append(course)
        
    for req_course in all_required_codes:
        code = req_course['รหัส'].strip('*')
        if code not in course_records or course_records[code].get('grade') in NON_EARNING_GRADES:
            unmet_requirements.append(f"ยังไม่ได้เรียนวิชาบังคับ: {code} {req_course['ชื่อไทย']}")

    gen_ed_prefixes = ('011', '012', '013', '0140', '0142', '01999', '02999')
    specific_prefixes = ('01418', '01417')
    gen_ed_credits_earned = sum(int(info.get('credits', 0)) for code, info in course_records.items() if info.get('grade') not in NON_EARNING_GRADES and code.startswith(gen_ed_prefixes))
    specific_credits_earned = sum(int(info.get('credits', 0)) for code, info in course_records.items() if info.get('grade') not in NON_EARNING_GRADES and code.startswith(specific_prefixes))
    
    # General check
    gen_ed_credits_rule = _extract_credits(structure.get('หมวดวิชาศึกษาทั่วไป', {}).get('หน่วยกิต', '0'), 30)
    if gen_ed_credits_earned < gen_ed_credits_rule:
        unmet_requirements.append(f"ขาดหน่วยกิตในหมวดวิชาศึกษาทั่วไป {gen_ed_credits_rule - gen_ed_credits_earned} หน่วยกิต")

    # Specific check
    specific_credits_rule = _extract_credits(structure.get('หมวดวิชาเฉพาะ', {}).get('หน่วยกิต', '0'), 88)
    if specific_credits_earned < specific_credits_rule:
        unmet_requirements.append(f"ขาดหน่วยกิตในหมวดวิชาเฉพาะ {specific_credits_rule - specific_credits_earned} หน่วยกิต")

    # Free Elective check
    free_elective_credits_rule = _extract_credits(structure.get('หมวดวิชาเลือกเสรี', {}).get('หน่วยกิต', '0'), 6)
    # ต้องหักลบจากยอดรวมทั้งหมด
    free_elective_credits_earned = final_total_credits - (gen_ed_credits_earned + specific_credits_earned)
    if free_elective_credits_earned < free_elective_credits_rule:
        unmet_requirements.append(f"ขาดหน่วยกิตในหมวดวิชาเลือกเสรี {free_elective_credits_rule - free_elective_credits_earned} หน่วยกิต")
    
    # Activity hours check
    activity_result = _check_activities(activity_data)
    if activity_result["unmet_requirement"]:
        unmet_requirements.append(activity_result["unmet_requirement"])

    # Payment check
    
    latest_term_name = "Unknown"
    if semester_gpas:
        # พยายามหาเทอมสุดท้ายที่มีเกรด (หรือเทอมปัจจุบัน)
        latest_term_name = semester_gpas[-1].get('term') or semester_gpas[-1].get('semester_full_name')

    payment_info = {
        "status_clear": payment_status_clear,
        "amount": payment_amount,
        "date": payment_date,
        "channel": payment_channel,
        "term_year_semester": payment_term_year_semester
    }
    
    # เรียก _check_payment โดยส่ง latest_term_name เข้าไปเทียบ
    payment_result = _check_payment(payment_info, latest_term_name)
    if payment_result["unmet"]:
        unmet_requirements.append(payment_result["unmet"])
    
    courses_by_semester = defaultdict(list)
    for course in transcript_data:
        semester_key = course.get('semester_full_name', 'Unknown')
        courses_by_semester[semester_key].append(course)

    semesters_summary = {}
    term_stats_map = {}
    for item in semester_gpas:
        semester_name = item.get('term') or item.get('semester_full_name')
        if semester_name:
            term_stats_map[semester_name] = item
            
    sorted_semesters = sorted(courses_by_semester.keys(), key=lambda s: (s.split('(')[-1].strip(')'), s))
        
    for semester in sorted_semesters:
        term_courses = courses_by_semester[semester]
        
        stats = term_stats_map.get(semester, {})
        semesters_summary[semester] = {
            "courses": term_courses,
            "term_gpa": f"{stats.get('term_gpa', 0.0):.2f}",
            "cumulative_gpa": f"{stats.get('cumulative_gpa', 0.0):.2f}"
        }
    
    # Summary
    is_eligible = not unmet_requirements
    overall_summary_message = "คุณสมบัติเบื้องต้นครบถ้วนสำหรับการสำเร็จการศึกษา" if is_eligible else "ยังไม่ผ่านเกณฑ์การสำเร็จการศึกษา"

    return {
        "student_id": student_id,
        "student_name": student_name,
        "is_eligible_for_graduation": is_eligible,
        "overall_summary_message": overall_summary_message,
        "cumulative_gpa": gpa_display,
        "total_credits_earned": real_credits,
        "unmet_requirements": list(set(unmet_requirements)), 
        "semesters_summary": semesters_summary,
        "activity_summary": {
            "status_text": activity_result["status"],
            "details": activity_result["details"]
        },
        "payment_summary": {
            "status_text": payment_result["status"],
            "details": payment_result["details"]
        }
    }