
import random
import numpy as np

moves = [(1,0),(0,1),(-1,0),(0,-1)]

def createRandomAreas(width, height, areas):
    remainingcells = width*height
    existing = 0
    
    while existing<areas:    
        area_size = random.randint(remainingcells//(2*areas), (remainingcells*3//2//areas))
        nodes = [(0,0)]
        visited = set(nodes)
        while nodes:
            node = nodes.pop(0)
            cx,cy = node
            
            for mx,my in random.sample(moves):
                nx = cx+mx
                ny = cy+my
                if (nx,ny) not in visited and 0<=nx<width and 0<=ny<height:
                    nodes.append((nx,ny))
                
            area_size -= 1
            if area_size<=0:
                break
        
                    

            


def main():
    pass

if __name__ == '__main__':
    main()
