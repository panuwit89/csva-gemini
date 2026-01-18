import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from dateutil.parser import parse as parse_date
from collections import defaultdict
from google.genai import types
from . import global_state

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
                    "transcript_data": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.OBJECT)),
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

def get_current_academic_term(check_date: datetime):
    """Determines the academic term based on the given date."""
    year = check_date.year
    month = check_date.month
    buddhist_year = year + 543

    if month >= 11 or month <= 3:
        academic_year = buddhist_year if month >= 11 else buddhist_year - 1
        return "ภาคปลาย", str(academic_year)
    elif 4 <= month <= 5:
        return "ภาคฤดูร้อน", str(buddhist_year - 1)
    else: 
        return "ภาคต้น", str(buddhist_year)

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

def _check_payment(payment_info: Dict) -> Dict:
    is_paid_successfully = (
        payment_info.get("amount") is not None and payment_info["amount"] > 0 and
        payment_info.get("date") is not None and
        payment_info.get("channel") is not None
    )

    if not is_paid_successfully:
        return {
            "status": "ตรวจสอบไม่ได้",
            "details": "ไม่สามารถสกัดข้อมูลสำคัญ (ยอดชำระ, วันที่, ช่องทาง) จากใบเสร็จได้",
            "unmet_requirement": "ไม่สามารถตรวจสอบข้อมูลการชำระเงินจากใบเสร็จได้ กรุณาตรวจสอบไฟล์อีกครั้ง"
        }

    unmet = []
    details = f"จากใบเสร็จ KU2 ที่คุณแนบมา ยอดชำระล่าสุดคือ {payment_info.get('amount')} บาท ชำระเมื่อวันที่ {payment_info.get('date')} ผ่าน{payment_info.get('channel')} สำหรับ{payment_info.get('term_year_semester', 'N/A')}"
    latest_term_message = ""
    receipt_term_text = payment_info.get("term_year_semester")
    
    if receipt_term_text:
        try:
            current_term, current_year = get_current_academic_term(datetime.now())
            
            # ใช้ regex เพื่อหาปีและภาคการศึกษาอย่างยืดหยุ่น
            term_pattern = r'(ภาคต้น|ภาคปลาย|ภาคฤดูร้อน|First Semester|Second Semester|Summer Semester)'
            year_pattern = r'(\d{4})'

            term_match = re.search(term_pattern, receipt_term_text, re.IGNORECASE)
            year_match = re.search(year_pattern, receipt_term_text)

            if year_match and term_match:
                receipt_year_str = year_match.group(1)
                receipt_term_str = term_match.group(1)

                term_map = {
                    "first semester": "ภาคต้น", "ภาคต้น": "ภาคต้น",
                    "second semester": "ภาคปลาย", "ภาคปลาย": "ภาคปลาย",
                    "summer semester": "ภาคฤดูร้อน", "ภาคฤดูร้อน": "ภาคฤดูร้อน"
                }
                receipt_term = term_map.get(receipt_term_str.lower())

                # แปลงปี ค.ศ. เป็น พ.ศ. ถ้าจำเป็น
                receipt_year_int = int(receipt_year_str)
                if receipt_year_int < 2500: # สันนิษฐานว่าเป็นปี ค.ศ.
                    receipt_year = str(receipt_year_int + 543)
                else:
                    receipt_year = receipt_year_str

                if receipt_year == current_year and receipt_term == current_term:
                    latest_term_message = f"ซึ่งเป็นภาคการศึกษาปัจจุบัน ({current_term} {current_year}) ที่ถูกต้องสำหรับการยื่นจบ"
                else:
                    unmet.append(f"ใบเสร็จที่แนบมาไม่ใช่ของภาคการศึกษาปัจจุบัน ควรเป็นของ: {current_term} ปีการศึกษา {current_year}")
                    latest_term_message = f"ซึ่งไม่ใช่ภาคการศึกษาปัจจุบัน (ควรเป็น {current_term} {current_year})"
            else:
                raise ValueError("Could not find year and term in receipt text")
        
        except (ValueError, IndexError) as e:
            print(f"Error parsing receipt term '{receipt_term_text}': {e}")
            unmet.append("ไม่สามารถตรวจสอบความถูกต้องของภาคการศึกษาบนใบเสร็จได้")
            latest_term_message = "แต่ไม่สามารถตรวจสอบได้ว่าเป็นภาคการศึกษาปัจจุบันหรือไม่"
    else:
        unmet.append("ไม่พบข้อมูลภาคการศึกษาบนใบเสร็จ")
        latest_term_message = "แต่ไม่พบข้อมูลภาคการศึกษาบนใบเสร็จเพื่อทำการตรวจสอบ"

    return {
        "status": "ไม่ผ่านเกณฑ์" if unmet else "ผ่านเกณฑ์",
        "details": f"{details}. {latest_term_message}".strip(),
        "unmet_requirement": unmet[0] if unmet else None
    }

def check_graduation_status(
    student_id: str,
    student_name: str,
    transcript_data: List[Dict],
    final_cumulative_gpa: float,
    final_total_credits: int,
    semester_gpas: List[Dict],
    faculty: Optional[str] = None,
    field_of_study: Optional[str] = None,
    admission_year: Optional[int] = None,
    activity_data: Optional[List[Dict]] = None,
    payment_status_clear: Optional[bool] = None,
    payment_amount: Optional[float] = None,
    payment_date: Optional[str] = None,
    payment_channel: Optional[str] = None,
    payment_term_year_semester: Optional[str] = None
):
    if not transcript_data and not semester_gpas: 
        return { "error": "missing_documents", 
                "transcript_found": False, 
                "activity_found": bool(activity_data), 
                "payment_found": bool(payment_amount) 
                }
        
    curriculum_rules = global_state.curriculum_rules_json
    unmet_requirements = []
    gpa_display = f"{final_cumulative_gpa:.2f}" if final_cumulative_gpa is not None else "N/A"
    
    if not curriculum_rules:
        unmet_requirements.append("ไม่สามารถอ่านข้อมูลหลักสูตรได้")
        
    if not transcript_data and semester_gpas:
        print("Rebuilding transcript_data from semester_gpas...")
        rebuilt_transcript = []
        for semester in semester_gpas:
            if 'courses' in semester and isinstance(semester['courses'], list):
                semester_name = semester.get('term') or semester.get('semester_full_name', 'Unknown')
                for course in semester['courses']:
                    course['semester_full_name'] = semester_name
                    rebuilt_transcript.append(course)
        transcript_data = rebuilt_transcript
        print(f"Rebuilt {len(transcript_data)} courses into transcript_data.")
        
    if not transcript_data:
        unmet_requirements.append("ไม่สามารถอ่านข้อมูลรายวิชาจาก Transcript ได้")
    if not semester_gpas:
        unmet_requirements.append("ไม่สามารถอ่านข้อมูลเกรดเฉลี่ยแต่ละเทอมจาก Transcript ได้")

    # If critical data is missing, return a summary of what failed.
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
    
    # Transcript summary
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
    latest_transcript_term = sorted_semesters[-1] if sorted_semesters else None

    for semester in sorted_semesters:
        term_courses = courses_by_semester[semester]
        
        stats = term_stats_map.get(semester, {})
        semesters_summary[semester] = {
            "courses": term_courses,
            "term_gpa": f"{stats.get('term_gpa', 0.0):.2f}",
            "cumulative_gpa": f"{stats.get('cumulative_gpa', 0.0):.2f}"
        }
    
    # Use curriculum rules
    non_earning_grades = {'P', 'W', 'I', 'F', 'X', 'U', 'N'}
    min_total_credits_rule = int(curriculum_rules.get('หลักสูตร', {}).get('จำนวนหน่วยกิต', '0 124').split(' ')[-2])
    
    if final_cumulative_gpa is not None:
        if final_cumulative_gpa < 2.00:
            unmet_requirements.append(f"GPA สะสม ({gpa_display}) ต่ำกว่าเกณฑ์ขั้นต่ำ 2.00")
    else:
        unmet_requirements.append("ไม่สามารถคำนวณ GPA สะสมล่าสุดได้ (อาจมีเกรด N หรือ I ในเทอมสุดท้าย)")

    if final_total_credits < min_total_credits_rule:
        unmet_requirements.append(f"หน่วยกิตรวม ({final_total_credits}) ต่ำกว่าเกณฑ์ขั้นต่ำ {min_total_credits_rule} หน่วยกิต")
    
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
        if code not in course_records or course_records[code].get('grade') in non_earning_grades:
            unmet_requirements.append(f"ยังไม่ได้เรียนวิชาบังคับ: {code} {req_course['ชื่อไทย']}")

    gen_ed_prefixes = ('011', '012', '013', '0140', '0142', '01999', '02999')
    specific_prefixes = ('01418', '01417')
    gen_ed_credits_earned = sum(int(info.get('credits', 0)) for code, info in course_records.items() if info.get('grade') not in non_earning_grades and code.startswith(gen_ed_prefixes))
    specific_credits_earned = sum(int(info.get('credits', 0)) for code, info in course_records.items() if info.get('grade') not in non_earning_grades and code.startswith(specific_prefixes))
    
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
    payment_info = {
        "status_clear": payment_status_clear,
        "amount": payment_amount,
        "date": payment_date,
        "channel": payment_channel,
        "term_year_semester": payment_term_year_semester
    }
    payment_result = _check_payment(payment_info)
    if payment_result["unmet_requirement"]:
        unmet_requirements.append(payment_result["unmet_requirement"])
        
    # Summary
    is_eligible = not unmet_requirements
    overall_summary_message = "คุณสมบัติเบื้องต้นครบถ้วนสำหรับการสำเร็จการศึกษา" if is_eligible else "ยังไม่ผ่านเกณฑ์การสำเร็จการศึกษา"

    return {
        "student_id": student_id,
        "student_name": student_name,
        "is_eligible_for_graduation": is_eligible,
        "overall_summary_message": overall_summary_message,
        "cumulative_gpa": gpa_display,
        "total_credits_earned": final_total_credits,
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