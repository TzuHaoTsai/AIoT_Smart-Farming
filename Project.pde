import ddf.minim.*;

Minim myMinim;
AudioPlayer music_ready, music_game, music_lose;

PImage imgBG;
PImage start, earth, heart, sad_earth, restart;
PImage rock[] = new PImage[26];                //0~25(a~z)

int heart_times = 0;
boolean BG_flag = true;
boolean Using_keyboard = true;


void setup(){

  size(750,750);
  
  myMinim = new Minim(this);
  
  imgBG = loadImage("background.png"); 
  start = loadImage("start.png");
  earth = loadImage("earth.png");
  heart = loadImage("heart.png");
  restart = loadImage("restart.png");
  sad_earth = loadImage("sad_earth.png");
  for(int i=0 ; i<26 ; i++){                //rock : 60*60 dpi
    rock[i] = loadImage("r" + i + ".png");  //rx.png
  }
  
  music_ready = myMinim.loadFile("Lucid_Dreamer.wav");
  music_game = myMinim.loadFile("Alternate.wav");
  music_lose = myMinim.loadFile("Dana.wav");
  music_ready.play();
}

void draw(){
        image(imgBG,0,0); 
        image(earth,310,310); 
        
      if(BG_flag){
        image(start,310,500);
        textSize(80);
        text("Attack meteorites",35,200);
        
        if((320<mouseX && mouseX<430) && (500<mouseY && mouseY<630) && (mouseButton == LEFT)) {
             BG_flag = false;
             music_ready.close();
             music_game.play();
        }
      }
      else{
        Create_Left_Rock();
        Create_Right_Rock();
        Create_Top_Rock();
        Create_Bottom_Rock();
        
        textSize(40);
        String SCORE = "Score : " + str(score);
        text(SCORE, 20, 60);
        
        End_Condition();
      }
}

char Left_Revert , Right_Revert, Top_Revert, Bottom_Revert;
int score = 0;

void keyPressed() {
  
  char Left_Revert = Conversion(Left_word);
  char Right_Revert = Conversion(Right_word);
  char Top_Revert = Conversion(Top_word);
  char Bottom_Revert = Conversion(Bottom_word);
  
  if(Using_keyboard){
    if(key == Left_Revert || key == (Left_Revert-32)) {
      Left_x = -120;
      Left_word = int(random(26));
      Left_speed = random(3);
      
      score += 10;
    }
    else if(key == Right_Revert || key == (Right_Revert-32)) {
      Right_x = 810;
      Right_word = int(random(26));
      Right_speed = random(3);
      
      score += 10;
    }
    else if(key == Top_Revert || key == (Top_Revert-32)) {
      Top_y = -120;
      Top_word = int(random(26));
      Top_speed = random(3);
      
      score += 10;
    }
    else if(key == Bottom_Revert || key == (Bottom_Revert-32)) {
      Bottom_y = 810;
      Bottom_word = int(random(26));
      Bottom_speed = random(3);
     
      score += 10;
    }
  }
}

void mousePressed(){
  
  if(((350<mouseX && mouseX<410) && (500<mouseY && mouseY<560)) && (mouseButton == LEFT)){
    hit = 0;      
    score = 0;
    Using_keyboard = true;
    
    music_lose.rewind();
    music_lose.pause();
    
    music_game.rewind();
    music_game.play();
  }
}
