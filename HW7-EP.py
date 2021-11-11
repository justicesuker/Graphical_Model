# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:09:15 2021
HW7
@author: micha
The code is ugly, but it works so who cares...
"""
import VE

# Input the known factors 1:A,2:B,3:C,4:D
phi1 = VE.Factor([1,2],{(0,0):10, (0,1):0.1, (1,0):0.1, (1,1):10})
phi2 = VE.Factor([1,3],{(0,0):5,  (0,1):0.2, (1,0):0.2, (1,1):5})
phi3 = VE.Factor([2,4],{(0,0):5,  (0,1):0.2, (1,0):0.2,  (1,1):5})
phi4 = VE.Factor([3,4],{(0,0):0.5, (0,1):1,  (1,0):20, (1,1):2.5})

# The factor list corresponding to two clusters
phi_C1 = [phi1]
phi_C2 = [phi2,phi3,phi4]

############################### Question 1 #######################################
############## First iteration: ########################
# Forward pass, m12
m12_iter1_2 = VE.VE(phi_C1,1)[0]
m12_iter1_1 = VE.VE(phi_C1,2)[0]

# Independent auumption.
m12_iter1 = VE.merge_factor([m12_iter1_2,m12_iter1_1])

# Nomalize
m12_iter1 = VE.normalize_factor(m12_iter1)
# Show the result
print(m12_iter1.value)

# Calculate the belief of C_2:
psi_c2_iter1 = [m12_iter1,phi2,phi3,phi4]

# Show the result on the slides:
psi_c2_iter1_34 = VE.sum_product_eliminate_var(psi_c2_iter1,[1,2])
psi_c2_iter1_34 = VE.merge_factor(psi_c2_iter1_34)
psi_c2_iter1_34 = VE.normalize_factor(psi_c2_iter1_34)
print(psi_c2_iter1_34.value)

# Backward pass:
m21_iter1_1 = VE.sum_product_eliminate_var(psi_c2_iter1,[2,3,4])[0]
m21_iter1_1 = VE.normalize_factor(m21_iter1_1)
print(m21_iter1_1.value)
m21_iter1_2 = VE.sum_product_eliminate_var(psi_c2_iter1,[1,3,4])[0]
m21_iter1_2 = VE.normalize_factor(m21_iter1_2)
print(m21_iter1_2.value)    

# Show the result on the slides for P(A,B):
psi_c1_iter1 = VE.merge_factor([m21_iter1_1,m21_iter1_2,phi1])
psi_c1_iter1 = VE.normalize_factor(psi_c1_iter1)
print(psi_c1_iter1.value)

############## Second iteration: ########################
# Forward pass, m12
m12_iter2_2 = VE.VE([psi_c1_iter1],1)[0]
m12_iter2_1 = VE.VE([psi_c1_iter1],2)[0]

# Independent assumption.
m12_iter2 = VE.merge_factor([m12_iter2_2,m12_iter2_1])

# Nomalize
m12_iter2 = VE.normalize_factor(m12_iter2)
# Show the result
print(m12_iter2.value)

# Calculate the belief of C_2:
psi_c2_iter2 = [m12_iter2,phi2,phi3,phi4]

# Show the result on the slides:
psi_c2_iter2_34 = VE.sum_product_eliminate_var(psi_c2_iter2,[1,2])
psi_c2_iter2_34 = VE.merge_factor(psi_c2_iter2_34)
psi_c2_iter2_34 = VE.normalize_factor(psi_c2_iter2_34)
print(psi_c2_iter2_34.value)

# Backward pass:
m21_iter2_1 = VE.sum_product_eliminate_var(psi_c2_iter2,[2,3,4])[0]
m21_iter2_1 = VE.normalize_factor(m21_iter2_1)
print(m21_iter2_1.value)
m21_iter2_2 = VE.sum_product_eliminate_var(psi_c2_iter2,[1,3,4])[0]
m21_iter2_2 = VE.normalize_factor(m21_iter2_2)
print(m21_iter2_2.value)    

# Show the result on the slides for P(A,B):
psi_c1_iter2 = VE.merge_factor([m21_iter2_1,m21_iter2_2,phi1])
psi_c1_iter2 = VE.normalize_factor(psi_c1_iter2)
print(psi_c1_iter2.value)


############################### Question 2 #######################################
############## First iteration: ########################
# We simply use the previous result to calculate the 1st iteration:
# m12_iter1
# m21_iter1 
m12_iter1_ep1 = VE.VE([psi_c1_iter1],2)[0]
m12_iter1_ep2 = VE.VE([psi_c1_iter1],1)[0]

m12_iter1_ep1 = VE.ep_approx(m12_iter1_ep1, m21_iter1_1)
print(m12_iter1_ep1.value)

m12_iter1_ep2 = VE.ep_approx(m12_iter1_ep2, m21_iter1_2)
print(m12_iter1_ep2.value)

# Calculate the belief of C_2:
# Independent assumption.
m12_iter1_ep = VE.merge_factor([m12_iter1_ep1,m12_iter1_ep2])

# Nomalize
m12_iter1_ep = VE.normalize_factor(m12_iter1_ep)
    
psi_c2_iter1_ep = [m12_iter1_ep,phi2,phi3,phi4]

# Show the result on the slides:
psi_c2_iter1_ep34 = VE.sum_product_eliminate_var(psi_c2_iter1_ep,[1,2])
psi_c2_iter1_ep34 = VE.merge_factor(psi_c2_iter1_ep34)
psi_c2_iter1_ep34 = VE.normalize_factor(psi_c2_iter1_ep34)
print(psi_c2_iter1_ep34.value)

# backward pass:
m21_iter1_ep1 = VE.sum_product_eliminate_var(psi_c2_iter1_ep,[2,3,4])[0]
m21_iter1_ep2 = VE.sum_product_eliminate_var(psi_c2_iter1_ep,[1,3,4])[0]

m21_iter1_ep1 = VE.ep_approx(m21_iter1_ep1, m12_iter1_ep1)
print(m21_iter1_ep1.value)

m21_iter1_ep2 = VE.ep_approx(m21_iter1_ep2, m12_iter1_ep2)
print(m21_iter1_ep2.value)

# Calculate the belief of C_1:
# Independent assumption.
psi_c1_iter1_ep = VE.merge_factor([m21_iter1_ep1, m21_iter1_ep2, phi1])
psi_c1_iter1_ep = VE.normalize_factor(psi_c1_iter1_ep)
print(psi_c1_iter1_ep.value)

############## Seond iteration: ########################
m12_iter2_ep1 = VE.VE([psi_c1_iter1_ep],2)[0]
m12_iter2_ep2 = VE.VE([psi_c1_iter1_ep],1)[0]

m12_iter2_ep1 = VE.ep_approx(m12_iter2_ep1, m21_iter1_ep1)
print(m12_iter2_ep1.value)

m12_iter2_ep2 = VE.ep_approx(m12_iter2_ep2, m21_iter1_ep2)
print(m12_iter2_ep2.value)

# Calculate the belief of C_2:
# Independent assumption.
m12_iter2_ep = VE.merge_factor([m12_iter2_ep1, m12_iter2_ep2])

# Nomalize
m12_iter2_ep = VE.normalize_factor(m12_iter2_ep)
    
psi_c2_iter2_ep = [m12_iter2_ep,phi2,phi3,phi4]

# Show the result on the slides:
psi_c2_iter2_ep34 = VE.sum_product_eliminate_var(psi_c2_iter2_ep,[1,2])
psi_c2_iter2_ep34 = VE.merge_factor(psi_c2_iter2_ep34)
psi_c2_iter2_ep34 = VE.normalize_factor(psi_c2_iter2_ep34)
print(psi_c2_iter2_ep34.value)

# backward pass:
m21_iter2_ep1 = VE.sum_product_eliminate_var(psi_c2_iter2_ep,[2,3,4])[0]
m21_iter2_ep2 = VE.sum_product_eliminate_var(psi_c2_iter2_ep,[1,3,4])[0]

m21_iter2_ep1 = VE.ep_approx(m21_iter2_ep1, m12_iter2_ep1)
print(m21_iter2_ep1.value)

m21_iter2_ep2 = VE.ep_approx(m21_iter2_ep2, m12_iter2_ep2)
print(m21_iter2_ep2.value)

# Calculate the belief of C_1:
# Independent assumption.
psi_c1_iter2_ep = VE.merge_factor([m21_iter2_ep1, m21_iter2_ep2, phi1])
psi_c1_iter2_ep = VE.normalize_factor(psi_c1_iter2_ep)
print(psi_c1_iter2_ep.value)

