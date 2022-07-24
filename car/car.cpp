#include<graphics.h>
#include<iostream>
using namespace std;
int main()
{
initwindow(1200,800);
int i=0,j,midy;
midy=getmaxy()/2;
pieslice(0,500,0,90,30);
while(1)
{
	i++;
	if(i>1000)
	{
		i=0;
	}
	setcolor(YELLOW);
	setfillstyle(SOLID_FILL,YELLOW);
	pieslice(0+i,500,0-i,90-i,30);
	pieslice(0+i,500,180-i,270-i,30);
	circle(0+i,500,30);
	arc(0+i,500,0,180,40);
	
	pieslice(300+i,500,0-i,90-i,30);
	pieslice(300+i,500,180-i,270-i,30);
	circle(300+i,500,30);
	arc(300+i,500,0,180,40);
	
	line(40+i,500,260+i,500);
	line(-200+i,500,-40+i,500);
	line(340+i,500,540+i,500);
	line(-200+i,500,-200+i,400);
	line(-200+i,400,-110+i,380);
	line(-110+i,380,-30+i,270);
		
	line(-30+i,270,210+i,270);
	line(210+i,270,320+i,380);
	line(320+i,380,500+i,410);
	line(500+i,410,540+i,500);
	
	setcolor(BLACK);
	setfillstyle(SOLID_FILL,BLACK);
	
	line(-90+i,380,-20+i,280);
	line(-20+i,280,80+i,280);
	line(80+i,280,80+i,380);
	line(-90+i,380,80+i,380);
	floodfill(30+i,midy-20,BLACK);
	line(200+i,280,300+i,380);
	line(100+i,280,200+i,280);
	//line(100)
	line(100+i,280,100+i,380);
	line(100+i,380,300+i,380);
	floodfill(230+i,midy-20,BLACK);
	

	delay(20);
	cleardevice();
}
getch();
closegraph();
return 0;	
}
