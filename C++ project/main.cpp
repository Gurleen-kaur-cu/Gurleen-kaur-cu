#include<graphics.h>
#include<iostream>
using namespace std;
#include<unistd.h>
#include<stdlib.h>
#include<stdio.h>
#include<conio.h>
#include<dos.h>
#include <windows.h>
int main();
void money();
char lifelines_action();
void lifeline_half();
void switch_the_question();
void q1();
void q2();
void q3();
void q4();
void q5();
void q6();
void q7();
void q8();
void q9();
void q10();
void q11();
void q12();
void q13();
void q14();
void q15();
void q16();
int lifelie_q1();
int lifeline_q2();
int lifeline_q3();
int lifeline_q4();
int lifeline_q5();
int lifeline_q6();
int lifeline_q7();
int lifeline_q8();
int lifeline_q9();
int lifeline_q10();
int lifeline_q11();
int lifeline_q12();
int lifeline_q13();
int lifeline_q14();
int lifeline_q15();
int lifeline_q16();
int exit();
void restart();

void instructions()
{
	//box size
	line(50,50,1200,50);
	line(50,650,1200,650);
	line(50,50,50,650);
	line(1200,50,1200,650);
	setcolor(12);
	settextstyle(1,0,5);
	outtextxy(350,70,(char*) "INSTRUCTIONS");
	setcolor(5);
	settextstyle(1,0,1);
	outtextxy(80,140,(char*) "1)The questions will be multiple choice based.");
	outtextxy(80,170,(char*) "2)TIME ALLOCATION -->");
	setcolor(3);
	outtextxy(120,200,(char*) "No time limits");
	setcolor(5);
	outtextxy(80,230,(char*) "3)First question will be of rs.1000 after that the award will be doubled respectively.");
	outtextxy(80,260,(char*) "4)There will be three stages where your award will be locked");
	setcolor(3);
	outtextxy(120,290,(char*) "a) Rs.10,000");
	outtextxy(120,320,(char*) "b) Rs.3,20,000");
	outtextxy(120,350,(char*) "c)JACKPOT!!!  7 CRORE");
	setcolor(5);
	outtextxy(80,380,(char*) "5)Two LIFELINES are available");
	setcolor(3);
	outtextxy(120,410,(char*) "a)50:50 -->Reduces options to half");
	outtextxy(120,440,(char*) "b)Switch -->Changes the question");
	setcolor(5);
	outtextxy(80,470,(char*) "6)You can only use one lifeline per question.");
	outtextxy(80,500,(char*) "7)Press--> 1 for 50:50");
	outtextxy(80,530,(char*) "8)Press--> 2 for Switch the question");
	outtextxy(80,560,(char*) "9)Press--> 3 for ensuring that you do not need any lifeline");
	settextstyle(1,0,5);
	setcolor(8);
	outtextxy(450,600,(char*) "!!LET'S PLAY!!");
	delay(5000);//5 SECONDS
	cleardevice();	
}
void money()
{
  setcolor(5);
  line(850,20,850,500);	
  delay(550);
  line(1050,20,1050,500);
  delay(550);
  line(850,20,1050,20);
  delay(550);	
  line(850,500,1050,500);
  delay(550);
  setcolor(YELLOW);
  settextstyle(1,0,2);                                    
  outtextxy(865,28,(char*) "* 70,000,000"); 
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,58,(char*) "* 10,000,000");  
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,88,(char*) "* 50,00,000");   
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,118,(char*) "* 25,00,000");  
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,148,(char*) "* 12,50,000");  
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,178,(char*) "* 6,40,000"); 
  setcolor(YELLOW);
  settextstyle(1,0,2);
  outtextxy(865,208,(char*) "* 3,20,000"); 
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,238,(char*) "* 1,60,000"); 
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,268,(char*) "* 80,000");
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,298,(char*) "* 40,000");
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,328,(char*) "* 20,000");
  setcolor(YELLOW);
  settextstyle(1,0,2);
  outtextxy(865,358,(char*) "* 10,000");
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,388,(char*) "* 5,000");
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,418,(char*) "* 3,000");
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,448,(char*) "* 2,000");
  setcolor(BLUE);
  settextstyle(1,0,2);
  outtextxy(865,478,(char*) "* 1,000");
  delay(2000);
}
void lifeline_half()
{
  ellipse(1150,60,0,360,60,40);
  setcolor(2);
  outtextxy(1103,50,(char*) "1.50:50");
  delay(1000);
}
void switch_the_question()
{
	setcolor(BLUE);
	ellipse(1150,260,0,360,87,30);
	setcolor(2);
	outtextxy(1093,250,(char*) "2.Switch");
	delay(1000);
}
int exit()
{
	cleardevice();
	settextstyle(1,0,6);
	outtextxy(100,400,(char*)"Exiting game...");
	cleardevice();
	return 0;
}
void layout_options()
{
  setcolor(8);
  line(100,400,370,400);	
  line(95,405,95,495);
  line(375,405,375,495);
  line(100,500,370,500);
  
  line(420,400,690,400);	
  line(415,405,415,495);
  line(420,500,690,500);
  line(695,405,695,495);
  
  line(100,550,370,550);	
  line(95,555,95,645);
  line(375,555,375,645);
  line(100,650,370,650);
  
  line(420,550,690,550);	
  line(415,555,415,645);
  line(420,650,690,650);
  line(695,555,695,645);
}
void layout_question()
{
  setcolor(BLUE);
  line(100,150,700,150);
  line(100,350,700,350);
  line(95,155,20,247);
  line(20,252,95,345);
  line(705,155,780,247);
  line(780,252,705,345);
}
void restart()
{
	
		cleardevice();
		settextstyle(1,0,6);
		setcolor(RED);
		outtextxy(100,300,(char*)"You LOOSE");
		delay(2000);
		settextstyle(1,0,3);
		setcolor(9);
		outtextxy(100,450,(char*)"Do you want to restart the Game if YES then type 'Y' else press any key");
		delay(3000);
		char choose;
		cin>>choose;
		if(choose=='Y'||choose=='y')
		{
			cleardevice();
		main();
     	}
		else
		{
			exit();
		}
}
void lifeline_action()
{
	setcolor(6);
	outtextxy(30,30,(char*) "Do you need any lifeline? If yes then ");
	outtextxy(30,60,(char*) "Enter the sequenced integer");
	outtextxy(30,90,(char*) "Else type any key");
}
int lifeline_q1()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,555,365,645);
		line(365,555,105,645);
		line(425,555,685,645);
		line(685,555,425,645);
	}
	else if(life==2)
	{
	    char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(100,235,(char*) "How many time zones are there in Russia?");
		setcolor(10);
		outtextxy(150,435,(char*) "(A) 11");
		outtextxy(150,585,(char*) "(C) 10");
		outtextxy(430,435,(char*) "(B) 5");
		outtextxy(430,585,(char*) "(D) 15");
		cin>>ans;
		if(ans=='A'||ans=='a')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.1000");
    	delay(5000);
    	cleardevice();		
		}
		else
		{		
		restart();	
		}
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
		return life;
}
int lifeline_q2()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,555,365,645);
		line(365,555,105,645);
		line(425,405,685,495);
		line(685,405,425,495);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(100,235,(char*) "What’s the national flower of Japan?");
		setcolor(10);
		outtextxy(150,435,(char*) "(A) Rose");
		outtextxy(105,585,(char*) "(C) Cherry blossom");
		outtextxy(430,435,(char*) "(B) Lotus");
		outtextxy(430,585,(char*) "(D) Lily");
		cin>>ans;
		if(ans=='C'||ans=='c')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.2000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q3()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(415,555,685,645);
		line(685,555,415,645);
		line(105,405,365,495);
		line(365,405,105,495);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(85,235,(char*) "How many stripes are there on the US flag?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) 10");
		outtextxy(105,585,(char*) "(C) 11");
		outtextxy(430,435,(char*) "(B) 12");
		outtextxy(430,585,(char*) "(D) 13");
		cin>>ans;
		if(ans=='D'||ans=='d')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.3000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q4()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(425,405,685,495);
		line(685,405,425,495);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(100,235,(char*) "What’s the national animal of Australia?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) Red Kangaroo");
		outtextxy(105,585,(char*) "(C) Panda");
		outtextxy(430,435,(char*) "(B) White tiger");
		outtextxy(430,585,(char*) "(D) Zebra");
		cin>>ans;
		if(ans=='A'||ans=='a')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.5000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q5()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(105,555,365,645);
		line(365,555,105,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(60,235,(char*) "How many days does it take for the Earth to orbit");
		outtextxy(160,265,(char*) "the Sun?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) 366");
		outtextxy(105,585,(char*) "(C) 265");
		outtextxy(430,435,(char*) "(B) 365");
		outtextxy(430,585,(char*) "(D) 355");
		cin>>ans;
		if(ans=='B'||ans=='b')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.10,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q6()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(415,405,685,495);
		line(685,405,415,495);
		line(105,555,365,645);
		line(365,555,105,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(90,235,(char*) "Until 1923, what was the Turkish city of ");
		outtextxy(170,265,(char*) "Istanbul called?");
		setcolor(10);
		outtextxy(105,420,(char*) "(A) Ancient ");
		outtextxy(105,460,(char*) "Byzantium");
		outtextxy(105,585,(char*) "(C) Constantinople");
		outtextxy(425,420,(char*) "(B) Byzantine ");
		outtextxy(425,460,(char*) "Empire");
		outtextxy(425,560,(char*) "(D) Ottoman ");
		outtextxy(425,600,(char*) "Empire");
		cin>>ans;
		if(ans=='C'||ans=='c')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.20,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q7()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(415,405,685,495);
		line(685,405,415,495);
		line(415,555,685,645);
		line(685,555,415,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(100,235,(char*) "What country has the most islands in the world?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) Sweden");
		outtextxy(105,585,(char*) "(C) New Zealand");
		outtextxy(420,435,(char*) "(B) Singapore");
		outtextxy(430,585,(char*) "(D) Jamaica");
		cin>>ans;
		if(ans=='A'||ans=='a')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.40,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q8()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,555,365,645);
		line(365,555,105,645);
		line(415,555,685,645);
		line(685,555,415,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(150,235,(char*) "What’s the capital of Canada?");
		setcolor(10);
		outtextxy(150,435,(char*) "(A) Brampton");
		outtextxy(150,585,(char*) "(C) Toronto");
		outtextxy(430,435,(char*) "(B) Ottawa");
		outtextxy(430,585,(char*) "(D) Alberta");
		cin>>ans;
		if(ans=='B'||ans=='b')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.80,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q9()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(415,555,685,645);
		line(685,555,415,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(60,235,(char*) "Name the largest (not highest) mountain range in");
		outtextxy(160,265,(char*) "the world?");
		setcolor(10);
		outtextxy(120,435,(char*) "(A) The Andes");
		outtextxy(105,585,(char*) "(C) Mount Everest");
		outtextxy(430,435,(char*) "(B) K2");
		outtextxy(430,585,(char*) "(D) Lhotse");
		cin>>ans;
		if(ans=='A'||ans=='a')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.1,60,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q10()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(415,405,685,495);
		line(685,405,415,495);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(70,235,(char*) "Where is the lowest natural place on planet Earth?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) Pulpit Rock");
		outtextxy(105,585,(char*) "(C) Jeita Grotto");
		outtextxy(416,435,(char*) "(B) Mud volcanoes");
		outtextxy(417,560,(char*) "(D) The Mariana ");
		outtextxy(430,600,(char*) "Trench");
		cin>>ans;
		if(ans=='D'||ans=='d')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.3,20,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q11()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(105,555,365,645);
		line(365,555,105,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(150,235,(char*) "Name Disney’s first film? ");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) Cinderella");
		outtextxy(105,585,(char*) "(C) Snow White,");
		outtextxy(430,435,(char*) "(B) Fairy");
		outtextxy(430,585,(char*) "(D) Aladin");
		cin>>ans;
		if(ans=='C'||ans=='c')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.6,40,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q12()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,555,365,645);
		line(365,555,105,645);
		line(425,405,685,495);
		line(685,405,425,495);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(130,235,(char*) "What’s the smallest country in the world?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) The Vatican");
		outtextxy(105,585,(char*) "(C) Nauru");
		outtextxy(430,435,(char*) "(B) Monaco");
		outtextxy(430,585,(char*) "(D) Tuvalu");
		cin>>ans;
		if(ans=='A'||ans=='a')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.12,50,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q13()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,555,365,645);
		line(365,555,105,645);
		line(425,555,685,645);
		line(685,555,425,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(150,235,(char*) "Where is Billie Eilish from?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) Canada");
		outtextxy(105,585,(char*) "(C) Los Angeles");
		outtextxy(430,435,(char*) "(B) London");
		outtextxy(430,585,(char*) "(D) Brazil");
		cin>>ans;
		if(ans=='C'||ans=='c')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.25,00,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q14()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(415,405,685,495);
		line(685,405,415,495);
		line(415,555,685,645);
		line(685,555,415,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(150,235,(char*) "What city do The Beatles come from?");
		setcolor(10);
		outtextxy(105,435,(char*) "(A) Moracco");
		outtextxy(105,585,(char*) "(C) Antartica");
		outtextxy(430,435,(char*) "(B) New york");
		outtextxy(420,585,(char*) "(D) Liverpool");
		cin>>ans;
		if(ans=='D'||ans=='d')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.50,00,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q15()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(415,405,685,495);
		line(685,405,415,495);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(120,235,(char*) "How many keys does a classic piano have?");
		setcolor(10);
		outtextxy(150,435,(char*) "(A) 77");
		outtextxy(150,585,(char*) "(C) 88");
		outtextxy(430,435,(char*) "(B) 99");
		outtextxy(430,585,(char*) "(D) 66");
		cin>>ans;
		if(ans=='c'||ans=='C')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.1,000,000");
    	delay(5000);
    	cleardevice();			
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
int lifeline_q16()
{
	int life;
	cin>>life;
	if(life==1)
	{
		setcolor(RED);
		line(1100,40,1200,90);
		line(1200,40,1100,90);
		line(105,405,365,495);
		line(365,405,105,495);
		line(105,555,365,645);
		line(365,555,105,645);
	}
	else if(life==2)
	{
		char ans;
		cleardevice();
	    money(); 
        lifeline_half();
	    switch_the_question();		    
	    setcolor(RED);
	    line(1090,240,1200,290);
	    line(1200,240,1090,290);
	    layout_question();
	    layout_options();
	    setcolor(8);
		outtextxy(100,235,(char*) "Name Pixar’s first feature-length movie? ");
		setcolor(10);
		outtextxy(105,420,(char*) "(A) Toy Story,");
		outtextxy(105,460,(char*) " 1995");
		outtextxy(105,560,(char*) "(C) Inside out");
		outtextxy(105,600,(char*) "1887");
		outtextxy(430,420,(char*) "(B) Soul");
		outtextxy(430,460,(char*) "1997");
		outtextxy(430,560,(char*) "(D) Onward");
		outtextxy(430,600,(char*) "1992");
		cin>>ans;
		if(ans=='A'||ans=='a')
		{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(5);
		outtextxy(140,200,(char*) "JACKPOT!!!! ");
		setcolor(9);
    	outtextxy(100,300,(char*)"You WON rs.70,00,000");
    	cleardevice();
					
		}
		else
		{		
		restart();	
		}		
	}
	else
	{
		setcolor(RED);
		outtextxy(800,570,(char*)"continue...");
	}
	return life;
}
void q1()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Which of these sports requires you to shout out a ");
	outtextxy(200,270,(char*) "word loudly during play?");
	setcolor(10);
	outtextxy(150,435,(char*) "(A) Ludo");
	outtextxy(470,435,(char*) "(B) Kho-kho");
	outtextxy(110,585,(char*) "(C) Playing cards");
	outtextxy(470,585,(char*) "(D) Chess");
	lifeline_action();
	int life_store;
	life_store = lifeline_q1();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='B'||ans=='b')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.1000");
	delay(5000);
	cleardevice();
	lable:
	     settextstyle(1,0,5);
	     setcolor(9);	
	     outtextxy(90,300,(char*) "Displaying next question in three seconds");
	     delay(3000);
	     cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q2()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(110,230,(char*) "Which of these spices is the smallest in size? ");
	setcolor(10);
	outtextxy(150,435,(char*) "(A) Ajwain");
	outtextxy(470,435,(char*) "(B) Jeera");
	outtextxy(150,585,(char*) "(C) Saunf");
	outtextxy(460,585,(char*) "(D) Methi seeds");
	int life_store;
	lifeline_action();
	life_store = lifeline_q2();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='A'||ans=='a')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	    outtextxy(100,300,(char*)"You WON rs.2000");
	    delay(5000);
	    cleardevice();
	lable:
	     settextstyle(1,0,5);
	     setcolor(9);
	     outtextxy(90,300,(char*) "Displaying next question in three seconds");
	     delay(3000);
	     cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q3()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(150,230,(char*) "Who is the CEO of google?");
	setcolor(10);
	outtextxy(102,435,(char*) "(A) Mark Zukerberg");
	outtextxy(470,435,(char*) "(B) Elon Musk");
	outtextxy(110,585,(char*) "(C) Sundar Pichai");
	outtextxy(450,585,(char*) "(D) Jeff Bezos");
	lifeline_action();
	int life_store;
	life_store = lifeline_q3();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='C'||ans=='c')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.3000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q4()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Which racing driver has won the British Grand");
	outtextxy(240,260,(char*) "Prix tournament?");
	setcolor(10);
	outtextxy(150,420,(char*) "(A) Michael");
	outtextxy(150,460,(char*) "Schumacher");
	outtextxy(470,420,(char*) "(B) Valtteri");
	outtextxy(470,460,(char*) "Bottas");
	outtextxy(150,560,(char*) "(C) Max ");
	outtextxy(150,600,(char*) "Verstappen");
	outtextxy(470,560,(char*) "(D) Lewis");
	outtextxy(470,600,(char*) "Hamilton");
	lifeline_action();
	int life_store;
	life_store = lifeline_q4();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='D'||ans=='d')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.5000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q5()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "If Jwala Gutta and Ashwini compete as partners,");
	outtextxy(140,260,(char*) " in which sport are they participating?");
	setcolor(10);
	outtextxy(150,435,(char*) "(A) Tennis");
	outtextxy(440,435,(char*) "(B) Table Tennis");
	outtextxy(150,585,(char*) "(C) Squash");
	outtextxy(470,585,(char*) "(D) Badminton");
	lifeline_action();
	int life_store;
	life_store = lifeline_q5();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='D'||ans=='d')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.10,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q6()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Which battle in 1757 marked the beginning of ");
	outtextxy(180,260,(char*) " British occupation in India?");
	setcolor(10);
	outtextxy(150,435,(char*) "(A) Plassey");
	outtextxy(470,435,(char*) "(B) Assaye");
	outtextxy(150,585,(char*) "(C) Buxar");
	outtextxy(470,585,(char*) "(D) Cuddalore");
	lifeline_action();
	int life_store;
	life_store = lifeline_q6();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='A'||ans=='a')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.20,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q7()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "What kind of creature was Bakasur, whom Kansa");
	outtextxy(145,270,(char*) "sent to kill Sri Krishna in his childhood? ");
	setcolor(10);
	outtextxy(150,435,(char*) "(A) Deer");
	outtextxy(470,435,(char*) "(B) Snake");
	outtextxy(150,585,(char*) "(C) Bird");
	outtextxy(470,585,(char*) "(D) Lizard");
	lifeline_action();
	int life_store;
	life_store = lifeline_q7();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='C'||ans=='c')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.40,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q8()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Who are the only married couple elected to");
	outtextxy(185,270,(char*) " the 16th Lok Sabha? ");
	setcolor(10);
	outtextxy(110,420,(char*) "(A) Sukhbir Singh-");
	outtextxy(80,460,(char*) "Harshimrat Kaur Badal");
	outtextxy(430,420,(char*) "(B) Pappu Yadav-");
	outtextxy(427,460,(char*) "Ranjeet Ranjan");
	outtextxy(120,560,(char*) "(C) Priya Ranjan-");
	outtextxy(103,600,(char*) "Deepa Dasmunshi");
	outtextxy(470,560,(char*) "(D) Prakash");
	outtextxy(427,600,(char*) "Brinda Karat");
	lifeline_action();
	int life_store;
	life_store = lifeline_q8();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='B'||ans=='b')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.80,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q9()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Whose autobiography is titled ‘Man of Everest’");
	outtextxy(135,260,(char*) "also published as ‘Tiger of the snows’?");
	setcolor(10);
	outtextxy(103,420,(char*) "(A) Sir Edmund");
	outtextxy(120,460,(char*) "Hillary");
	outtextxy(430,420,(char*) "(B) Tenzing ");
	outtextxy(430,460,(char*) "Norgay");
	outtextxy(110,560,(char*) "(C) George ");
	outtextxy(110,600,(char*) "Mallory");
	outtextxy(422,560,(char*) "(D)  Major H P S ");
	outtextxy(430,600,(char*) "Aluhwalia");
	lifeline_action();
	int life_store;
	life_store = lifeline_q9();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='b'||ans=='B')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.1,60,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q10()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,200,(char*) "Which institution has directed for appointment of");
	outtextxy(60,240,(char*) " woman IPS officer in every district of Kerala to ");
	outtextxy(150,280,(char*) "oversee POCSO related cases?");
	setcolor(10);
	outtextxy(103,420,(char*) "(A) Supreme Court");
	outtextxy(130,460,(char*) "of India");
	outtextxy(420,405,(char*) "(B) Ministry of");
	outtextxy(420,445,(char*) "Women & Child ");
	outtextxy(420,470,(char*) "Development");
	outtextxy(130,560,(char*) "(C) High Court ");
	outtextxy(130,600,(char*) "of Kerala");
	outtextxy(416,555,(char*) "(D) National");
	outtextxy(420,590,(char*) "Commission");
	outtextxy(417,620,(char*) "for Women");
	lifeline_action();
	int life_store;
	life_store = lifeline_q10();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='c'||ans=='C')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.3,20,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q11()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,200,(char*) "What is the name of the global carbon-neutral");
	outtextxy(60,240,(char*) "consortium formed by major companies");
	outtextxy(60,280,(char*) "including Microsoft, Nike and Mercedes-Benz?");
	setcolor(10);
	outtextxy(105,420,(char*) "(A) Global");
	outtextxy(105,460,(char*) "Carbon-Neutral");
	outtextxy(422,420,(char*) "(B) Transform");
	outtextxy(422,460,(char*) " to Net Zero");
	outtextxy(105,560,(char*) "(C) Free from");
	outtextxy(105,600,(char*) " Carbon");
	outtextxy(422,560,(char*) "(D) Zero Carbon");
	outtextxy(422,600,(char*) "World");
	lifeline_action();
	int life_store;
	life_store = lifeline_q11();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='B'||ans=='b')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.6,40,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q12()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Who has been appointed as the Managing ");
	outtextxy(100,260,(char*) "Director & CEO of Indian Overseas Bank?");
	setcolor(10);
	outtextxy(150,420,(char*) "(A) Padmaja ");
	outtextxy(150,460,(char*) "Chundru");
	outtextxy(422,435,(char*) "(B) A K Goel");
	outtextxy(150,560,(char*) "(C) Sankara");
	outtextxy(150,600,(char*) " Narayanan");
	outtextxy(422,560,(char*) "(D) Partha Pratim");
	outtextxy(430,600,(char*) " Sengupt");
	lifeline_action();
	int life_store;
	life_store = lifeline_q12();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='d'||ans=='D')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.12,50,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q13()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,200,(char*) "Which of the following battles is subject matter");
	outtextxy(60,240,(char*) "of Alha Khand, a 12th Century epic poetic work ");
	outtextxy(265,280,(char*) "in Hindi?");
	setcolor(10);
	outtextxy(150,420,(char*) "(A) Battle of");
	outtextxy(150,460,(char*) "Mandsaur");
	outtextxy(422,420,(char*) "(B) Battle of");
	outtextxy(422,460,(char*) "Mahoba");
	outtextxy(150,560,(char*) "(C) Battle of ");
	outtextxy(150,600,(char*) " Chittor");
	outtextxy(422,560,(char*) "(D) Battle of");
	outtextxy(430,600,(char*) "Tarain");
	lifeline_action();
	int life_store;
	life_store = lifeline_q13();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='b'||ans=='B')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.25,00,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q14()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(170,200,(char*) "Which day is observed as World");
	outtextxy(260,240,(char*) "standards day?");
	setcolor(10);
	outtextxy(150,435,(char*) "(A) Oct 14");
	outtextxy(422,435,(char*) "(B) June 26");
	outtextxy(150,585,(char*) "(C) Nov 15 ");
	outtextxy(422,585,(char*) "(D) Dec 2");
	lifeline_action();
	int life_store;
	life_store = lifeline_q14();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='a'||ans=='A')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.50,00,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}	
}
}
void q15()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(150,235,(char*) "Ghotokach in mahabharat was the son of?");
	setcolor(10);
	outtextxy(105,435,(char*) "(A) Yudhishthir");
	outtextxy(422,435,(char*) "(B) Arjuna");
	outtextxy(105,585,(char*) "(C) Bhima ");
	outtextxy(422,585,(char*) "(D) Duryodhana");
	lifeline_action();
	int life_store;
	life_store = lifeline_q15();
	if(life_store==2)
	goto lable;
	else if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='c'||ans=='C')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.10,000,000");
	delay(5000);
	cleardevice();
	lable:
	settextstyle(1,0,5);
	setcolor(9);
	outtextxy(90,300,(char*) "Displaying next question in three seconds");
	delay(3000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void q16()
{
	char ans;
	int store;
	money(); 
    lifeline_half();
	switch_the_question();	
	layout_question();
	layout_options();
	setcolor(8);
	outtextxy(60,230,(char*) "Which of the following was the first Indian");
	outtextxy(60,270,(char*) "state to issue photo identity cards to its voters?");
	setcolor(10);
	outtextxy(110,435,(char*) "(A) Tamil nadu");
	outtextxy(422,435,(char*) "(B) Rajasthan");
	outtextxy(110,585,(char*) "(C) West bengal ");
	outtextxy(422,585,(char*) "(D) Haryana");
	lifeline_action();
	int life_store;
	life_store = lifeline_q16();
	if(life_store==1||life_store==3)
	{
	cin>>ans;	
	if(ans=='d'||ans=='D')
	{
		cleardevice();
		settextstyle(1,0,6);
		setcolor(5);
		outtextxy(140,200,(char*) "JACKPOT!!!! ");
		setcolor(9);
	outtextxy(100,300,(char*)"You WON rs.70,00,000");
	delay(5000);
	cleardevice();
	}
	else
	{
		restart();
	}
}
}
void rec()
{	
//outer rectangle
line(200,100,1100,100);
line(200,400,1100,400);
line(200,100,200,400);
line(1100,100,1100,400);
setfillstyle(SOLID_FILL,3);
delay(550);
//inner recatngle
floodfill(500,150,BLUE);
	line(300,200,1000,200);
	line(300,300,1000,300);	
	line(300,200,300,300);
	line(1000,200,1000,300);	
setfillstyle(SOLID_FILL,8);
delay(550);
floodfill(500,150,BLUE);
delay(550);
	settextstyle(1,0,5);
	outtextxy(250,230,(char*) "INTIALIZING the game...");
	delay(2000);
	cleardevice();	
}
int main()
{	
	initwindow(1300,1000);//window size
	circle(575,335,300);//outer circle
	delay(1000);
	circle(575,335,235);//inner circle
	setfillstyle(SOLID_FILL,YELLOW);
	delay(1000);
	floodfill(575,335,WHITE);
	delay(1000);
	setcolor(GREEN);
	settextstyle(1,0,5);
	outtextxy(385,100,(char*) "K");    
    delay(750);
	outtextxy(490,50,(char*) "A");    
    delay(750);
	outtextxy(610,50,(char*) "U");    
    delay(750);
	outtextxy(710,100,(char*) "N");    
    delay(750);
    setcolor(GREEN);
	settextstyle(1,0,5);
    outtextxy(345,490,(char*) "B");       
    delay(750);
    outtextxy(420,550,(char*) "N");       
    delay(750);
    outtextxy(550,580,(char*) "E");       
    delay(750);
    outtextxy(685,540,(char*) "G");       
    delay(750);
    outtextxy(760,475,(char*) "A");       
    delay(750);
    setcolor(BLUE);
	settextstyle(1,0,5);
    outtextxy(375,300,(char*) "CROREPATI");
    delay(2000);
    cleardevice();//clearing the device
    rec();//calling rec function	
	instructions();//calling intsructions 
	q1();
	q2();
	q3();
	q4();
	q5();
	q6();
	q7();
	q8();
	q9();
	q10();
	q11();
	q12();
	q13();
	q14();
	q15();
	q16();
	getch();
}
