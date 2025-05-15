import torch


print("Hello")
print(torch.__version__)
if torch.cuda.is_available():
    print("its avaialble")
else:
    print("not available ")


print("")
print("")
print("")
print("") 


print("")
print("")
print("")
print("") 


# Binary Cross-Entropy Loss for scalar
def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8  # To prevent log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))

# Inputs
x = torch.tensor(6.7)  # Input feature
y = torch.tensor(0.0)  # True label (binary)

b= torch.tensor(0.8985,requires_grad=True)
w= torch.tensor(0.880,requires_grad=True)

b1= torch.tensor(1.0,requires_grad=True)
w1= torch.tensor(0.0757,requires_grad=True)


z= w*x+b

y_pred=torch.sigmoid(z)

z2=y_pred*w1+b1


y_pred1=torch.sigmoid(z2)

loss = binary_cross_entropy_loss(y_pred1,y)


print(loss)




loss.backward()




print("")
print("")
print("")
print("") 



print(w.grad)
print(b.grad)

print("")
print("")
print("")
print("") 

print(w1.grad)
print(b1.grad)


