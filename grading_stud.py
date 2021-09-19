n = int(input("Input size"))
grade = []
for i in  range(n):
    value = int(input())
    grade.append(value)

# print(grade)



def grading_student(grade):
    final_grade = []

    for i in grade:
        # print(3<3)
        print((i%5) < 3 )

        # if (i%5) < 3:
        #     for j in range(3):
        #         if i >= 38 and (((i+j)%5) == 0):
        #             final_grade.append(i+j)
        # elif ((i%5) == 3):
        #     final_grade.append(i)
            
    


        if (i%5) < 3:
            final_grade.append(i)
        elif (i%5) == 3 and i>=38: 
            print("Inside else")
            for j in range(3):
                if i >= 38 and (((i+j)%5) == 0):
                    final_grade.append(i+j)
        else:
            final_grade.append(i)
    return final_grade

final_grades = grading_student(grade)
print(final_grades)