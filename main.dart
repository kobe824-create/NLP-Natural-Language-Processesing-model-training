Class Animal{
  Void Sound(){
    print('Animal Sound');
  }
}

Class Dog extends Animal{
  @override
  Void Sound() {
    print('Dog barks');
  }
}

void main() {
  Animal myDog = Dog();
  
  myDog.sound 
}