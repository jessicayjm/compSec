import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

# # change context and answer
ca_x = np.array([5,10,15,20,25,30])
ca_y_total_acc = np.array([0.811, 0.777, 0.783, 0.794, 0.760, 0.777])
ca_y_correct_acc = np.array([0.952, 0.925, 0.918, 0.938, 0.911, 0.918])
ca_y_incorrect_acc = np.array([0.103, 0.034, 0.103, 0.069, 0.000, 0.069])


plt.plot(ca_x, ca_y_total_acc, label='total')
for i, txt in enumerate(ca_y_total_acc.tolist()):
    plt.annotate(txt, (ca_x[i]-1, txt+0.01))
plt.plot(ca_x, ca_y_correct_acc, label='instances possible to answer')
for i, txt in enumerate(ca_y_correct_acc.tolist()):
    plt.annotate(txt, (ca_x[i]-1, txt+0.01))
plt.plot(ca_x, ca_y_incorrect_acc, label='instances impossible to answer')
for i, txt in enumerate(ca_y_incorrect_acc.tolist()):
    plt.annotate(txt, (ca_x[i]-1, txt+0.01))
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend()
# plt.tight_layout()
plt.xlabel('# adversial examples')
plt.ylabel('accuracy')
plt.title('all_possible dataset')
plt.savefig("context_answer.pdf")

# change to random answers
# ca_x = np.array([10, 20, 30, 50, 80, 100, 150])
# ca_y_total_acc = np.array([0.857, 0.834, 0.840, 0.874, 0.874, 0.211, 0.171])
# ca_y_correct_acc = np.array([0.911, 0.884, 0.904, 0.945, 0.856, 0.103, 0.007])
# ca_y_incorrect_acc = np.array([0.586, 0.586, 0.517, 0.517, 0.966, 0.759, 1.0])


# plt.plot(ca_x, ca_y_total_acc, label='total')
# for i, txt in enumerate(ca_y_total_acc.tolist()):
#     plt.annotate(round(txt,2), (ca_x[i]-1, txt-0.04))
# plt.plot(ca_x, ca_y_correct_acc, label='instances possible to answer')
# for i, txt in enumerate(ca_y_correct_acc.tolist()):
#     plt.annotate(round(txt,2), (ca_x[i]-1, txt+0.01))
# plt.plot(ca_x, ca_y_incorrect_acc, label='instances impossible to answer')
# for i, txt in enumerate(ca_y_incorrect_acc.tolist()):
#     plt.annotate(round(txt,2), (ca_x[i]-1, txt+0.01))
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.legend()
# # plt.tight_layout()
# plt.xlabel('# adversial examples')
# plt.ylabel('accuracy')
# plt.title('random dataset')
# plt.savefig("random_answer.pdf")