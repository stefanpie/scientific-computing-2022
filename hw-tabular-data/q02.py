import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    db = {}
    data_dir = "./data/school_data/"
    db["schools"] = pd.read_csv(data_dir + "schools_complete.csv")
    db["students"] = pd.read_csv(data_dir + "students_complete.csv")
    return db

def compute_passing_rate_table(db, passing_score=70):
    schools = db["schools"]
    students = db["students"]

    students["pass_math"] = students["math_score"] >= passing_score
    students["pass_reading"] = students["reading_score"] >= passing_score

    student_data_agg = students.groupby("school").agg(
        {"pass_math": "mean", "pass_reading": "mean", "Student ID": "count", "reading_score": "mean", "math_score": "mean"}
    )
    student_data_agg = student_data_agg.rename(columns={"Student ID": "num_students"})
    student_data_agg = student_data_agg.rename(columns={"reading_score": "avg_reading", "math_score": "avg_math"})

    school_data_combined = pd.merge(schools, student_data_agg, left_on="name", right_on="school", how="left")

    school_data_combined["overall_passing_rate"] = (
        school_data_combined["pass_math"] + school_data_combined["pass_reading"]
    ) / 2
    school_data_combined["budget_per_student"] = school_data_combined["budget"] / school_data_combined["num_students"]

    school_data_combined = school_data_combined.sort_values(by="overall_passing_rate", ascending=False)

    return school_data_combined


def compute_passing_rate_table_by_grade_level(db, passing_score=70):
    students = db["students"]

    students["pass_math"] = students["math_score"] >= passing_score
    students["pass_reading"] = students["reading_score"] >= passing_score

    student_data_grouped = students.groupby(["school", "grade"])
    student_pass_data_agg = student_data_grouped.agg({"pass_math": "mean", "pass_reading": "mean"})
    student_pass_data_agg = student_pass_data_agg.rename(columns={"Student ID": "num_students"})

    student_pass_data_agg = student_pass_data_agg.reset_index()

    return student_pass_data_agg


if __name__ == "__main__":

    db = load_data()

    passing_rate_data = compute_passing_rate_table(db, passing_score=70)
    print("Schools sorted by overall passing rate:")
    print(passing_rate_data)
    print()

    passing_rate_by_grade_data = compute_passing_rate_table_by_grade_level(db, passing_score=70)
    print("Computed passing rate by school and grade level:")
    print(passing_rate_by_grade_data)

    # Plotting the data by grade
    g = sns.FacetGrid(passing_rate_by_grade_data, col="grade", col_wrap=2)
    g.map(sns.scatterplot, "pass_math", "pass_reading")
    plt.tight_layout()
    plt.savefig("passing_rate_by_grade.png")

    # Plotting relationship between passing rate and budget
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    sns.scatterplot(x="budget_per_student", y="avg_math", data=passing_rate_data, ax=axs[0,0])
    sns.scatterplot(x="budget_per_student", y="avg_reading", data=passing_rate_data, ax=axs[0,1])

    sns.scatterplot(x="budget_per_student", y="pass_math", data=passing_rate_data, ax=axs[1,0])
    sns.scatterplot(x="budget_per_student", y="pass_reading", data=passing_rate_data, ax=axs[1,1])

    axs[0, 0].set_title("Average Math Score vs. Budget")
    axs[0, 1].set_title("Average Reading Score vs. Budget")
    axs[1, 0].set_title("Average Math Pass Rate vs. Budget")
    axs[1, 1].set_title("Average Reading Pass Rate vs. Budget")

    plt.tight_layout()
    plt.savefig("passing_rate_vs_budget.png")