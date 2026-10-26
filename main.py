from Planner_Agent import get_planner_agent
from Researcher_Agent import get_researcher_agent
from Writer_Agent import get_writer_agent
from Editor_Agent import get_editor_agent
from Presentor_Agent import get_presentor_agent
import time

planner = get_planner_agent()
researcher = get_researcher_agent()
writer = get_writer_agent()
editor = get_editor_agent()
presenter = get_presentor_agent()

def run_auto_researcher(user_topic):
    writer_vault = []
    print(f"\n Planner is drafting the structure...")
    plan_output = planner.invoke({"input": user_topic})["output"]

    raw_text = plan_output
    mini_topics = [line for line in raw_text.split("\n") if "TOPIC:" in line]
    for section in mini_topics:
        title, outline = section.split("|")
        print(f"Feeding to Researcher -> Title: {title.strip()}, Task: {outline.strip()}")
        print(f"\n Researcher is gathering facts...")
        time.sleep(20)
        section_research = researcher.invoke({
            "input": f"Topic: {title}. Plan context: {outline}"
        })["output"]
        print(f"\n Writer is drafting section: {title}")
        mini_draft = writer.invoke({
            "input": f"Topic: {title} OUTLINE: {outline}\n\n RESEARCH: {section_research}"
        })["output"]
        writer_vault.append(f"SECTION TITLE: {title}\nCONTENT:\n{mini_draft}")
    
    all_mini_drafts = "\n\n-----SECTION BREAK-----\n\n".join(writer_vault)
    
    print(f"\n Editor is refining and connecting sections")
    refined_report = editor.invoke({
        "input": f"PLAN: {plan_output} \n\n ALL MINI-DRAFTS: {all_mini_drafts}"
    })["output"]

    print(f"\n[5] Presenter is creating the final document...")
    final_output = presenter.invoke({
        "input": f"PLAN: {plan_output} \n\n REFINED REPORT: {refined_report}"
    })["output"]

    return final_output

if __name__ == "__main__":
    user_query = input("Enter the topic to be researched")
    
    print(f"🚀 Starting Auto-Researcher for: {user_query}")
    final_report = run_auto_researcher(user_query)
    
    file_name = "final_research_report.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    print(f"\n✅ Mission Accomplished! Report saved to {file_name}")