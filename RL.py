######################
# Author: Te-Yuan Liu
######################

######################
# Import library
######################
import numpy as np

######################
# Define function
######################
def init_P(_w):
    w = _w
    P = np.zeros((100,100,4))
    for i in range(100):
        for k in range(4):
            # move right
            if k==0:
                if i==0:                            # corner
                    P[i,10,k] = 1 - w + w/4
                    P[i,0,k] = w/4 + w/4
                    P[i,1,k] = w/4
                elif i==9:
                    P[i,19,k] = 1 - w + w/4
                    P[i,9,k] = w/4 + w/4
                    P[i,8,k] = w/4
                elif i==90:
                    P[i,90,k] = 1 - w + w/4 + w/4
                    P[i,80,k] = w/4
                    P[i,91,k] = w/4
                elif i==99:
                    P[i,99,k] = 1 - w + w/4 + w/4
                    P[i,89,k] = w/4
                    P[i,98,k] = w/4
                elif i/10==9 and i%10>0 and i%10<9: # edge
                    P[i,i,k] = 1 - w + w/4
                    P[i,i-10,k] = w/4
                    P[i,i-1,k] = w/4
                    P[i,i+1,k] = w/4
                elif i/10==0 and i%10>0 and i%10<9:
                    P[i,i+10,k] = 1 - w + w/4
                    P[i,i,k] = w/4
                    P[i,i-1,k] = w/4
                    P[i,i+1,k] = w/4
                elif i%10==0 and i/10>0 and i/10<9:
                    P[i,i+10,k] = 1 - w + w/4
                    P[i,i-10,k] = w/4
                    P[i,i,k] = w/4
                    P[i,i+1,k] = w/4
                elif i%10==9 and i/10>0 and i/10<9:
                    P[i,i+10,k] = 1 - w + w/4
                    P[i,i-10,k] = w/4
                    P[i,i-1,k] = w/4
                    P[i,i,k] = w/4
                else:                               # non boundary
                    P[i,i+10,k] = 1 - w + w/4
                    P[i,i-10,k] = w/4
                    P[i,i-1,k] = w/4
                    P[i,i+1,k] = w/4


######################
# Main function
######################
def main():
    print("hello")
if __name__ == "__main__":
    main()
