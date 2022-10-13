#include "../nn.cpp"

#include <bits/stdc++.h>
#include <random>
#include <sys/time.h>
#include <utility>
#include <x86intrin.h>

#define rep(i, n) for (int i = 0; (i) < ((int) (n)); (i)++)
#define sz(v) ((int) ((v).size()))
#define all(v) (v).begin(), (v).end()
#define debug(v) { cerr << __LINE__ << ": " << (#v) << "=" << (v) << endl; }
#define debug2(v0, v1) { cerr << __LINE__ << ": " << (#v0) << "=" << (v0) << ", " << (#v1) << "=" << (v1) << endl; }
#define debug3(v0, v1, v2) { cerr << __LINE__ << ": " << (#v0) << "=" << (v0) << ", " << (#v1) << "=" << (v1) <<  ", " << (#v2) << "=" << (v2) << endl; }

using namespace std;
using namespace nn_eda;


float my_rand(float mini, float maxi)
{
  int mask = (1 << 25) - 1;
  float f = ((float) (rand() & mask)) / mask;
  return mini + (maxi - mini) * f;
}

vector<float> gen_input(int dim, int target)
{
  vector<float> res;
  rep(i, dim) res.push_back(my_rand(0, 2));
  res[target] = 1;
  return res;
}

vector<float> gen_label(int dim, int target)
{
  vector<float> res(dim, 0);
  res[target] = 1;
  return res;
}

double validate_model(Nn &nn, int dim, int n)
{
  srand(22222);

  double total_prob = 0.0;
  rep(loop, n){
    int target = rand() % dim;
    vector<float> input = gen_input(dim, target);
    total_prob += nn.forward(input)[target];
  }
  double avg_prob = total_prob / n;
  return avg_prob;
}

Nn train_model(int dim)
{
  Nn nn(vector<int>({dim, dim * 5, dim * 5, dim}));
  int num_epochs = 10;
  int batch_size = 16;
  int n = 10000;

  srand(2345);

  rep(epoch, num_epochs){
    // validation
    {
      double avg_prob = validate_model(nn, dim, n);
      debug2(epoch, avg_prob);
    }

    // train
    {
      rep(loop, n){
        vector<vector<float>> inputs;
        vector<vector<float>> labels;
        rep(i, batch_size){
          int target = rand() % dim;
          inputs.push_back(gen_input(dim, target));
          labels.push_back(gen_label(dim, target));
        }
        nn.train(inputs, labels, 0.01);
      }
    }
  }

  return nn;
}

int main()
{
  // 1. run local training and export the trained to codingame_nn.txt.
  // 2. import the model from codingame_nn.txt to codingame_test.cpp.
  // 3. run local validation.
  // 4. submit the code to codingame and run the validation.
  // 5. compare the outputs from 3 and 4.

  int dim = 5;
  int validation_n = 10000;
  #ifdef LOCAL
  bool is_local = true;
  #else
  bool is_local = false;
  #endif

  debug(is_local);

  if(is_local){
    Nn nn = train_model(dim);
    NnIo::to_obj(nn, 9).write("codingame_nn.txt");
    debug(validate_model(nn, dim, validation_n));
    // 111: validate_model(nn, dim, validation_n)=0.883459
  }else{
    Nn nn = nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(
      std::vector<int>({5, 25, 25, 5, }),
      "㈩㗇㈠犌橠㬦猌樤䔠獌訯뭒煌臟묾琌稘㤢猌榷뷬猈娨㤜騎騿뭌睌娰伩猌椇벉蟝뛪㬤뤢㨤㌢玅驀㬞燍䨼伯瑌㧻륄犌䩻뭄犊稨㜣珍騸㬝溧樘愫㻀㩌䔦犌娠㌣犌媰㰠猍舧봺焌穀㌥犊똩뵂瑋詠㴣猋뙂㔢牌橈㬧睌橏䔞狌娟덃燋뚂䌞猌樰㜡爋뙂㼤窌穦甯狌䨋鴮玌訸㤢琌㨜䤠牌稰䄟牋稧鬭琋穘㌧猌㩋봽牌詨㤦猌稰㜤猌訸㬤狌樰㴣炌樠䤣囍㨛紬猌㩔㬤犑婰㔦狌秈㴟熋訸䬛猋騠㔡灌婀䄤罌穽甪犌䨃謫王穀㌠琋訨㬢獋뙚㔣獊穇蜱燌橀㜠爌㨴㔠牌橏뭐獌機띂玌穀㔡犋驀㌡狌䩑㌧猋䫤㤢狌樷鼱珌婰㴥牌觀儯烍䨬㴐獌䨳祃犋똹郪玌摘㬠猌樸㤣灋騸䜡奌뙉蜭猌婐㤥犑㩤㔞狌秀㼡燌穀䜠犌䩤㤧烌橇뵁渌窈笭狍䨔㴟瀌穠䄔牌樿筄煋駿飬玌蒀㜡爌㨜䄢珌穐㔡玌訨㌥牋똹뵀爌䩓뵅珌詘㤟煋뙪㜦炌詠㜡牌㩌㔣犍㨼㼤爌計䬤犌㨴㤤灌䨳딾獋뙲㬟犌詈㜠焌뙊㜣燋稘唭爌樘㴞氌樸䄧猌媐㤟肻娰䄢猋評㤦牊竰㜠玌稯녃狌穨㴣爌㧬䬟牌樨㴠烌騏물煌驀㴞犌驇괼爊똱띁狌樸㴤狌穈㜥珌뙒㼧獌穟뵁爌㨤㼥猌㧼䴤獋똲㴠琋訰㔦猌䪄㴢剓婀䄟珌驀㌦燌橐㌦玌驘㜦狌娸㤠獌樸㼤猌訸㜥癋驀㴣爌樐㌠爊㨴㼦王窈㼣琌驧뵀王뙪㔣獌驟뵄狋뙪㼤玌驘㼥玌娘伪肋婸䌞犋뙊㼥猾窀㴢猌穐㼡狊詘㔦猌䨼䄓犌訸㼤縋㧻묵煌橘㬨皃㫔欼琊驈㔟梎檐圴牋穀㔮畎竸壺玌誇鴳犌樿땂璋穠㼨璋稰㔩爌䨓뵎盏䠤欽眂뛺洼烋뗱녧矎矐敁牌䩄㼤皏㫺꽍睏諾餼畊驨㔤澌誰䄣牌뙢㤢猌橈㤣悼姈㢂㵀榫렷低䉌㤊猳扯攭둞櫀㯞界橀鄫笋䩴䬥琋訯뵏栌䮕眮畎뚲㤣擑筰蓲㈠",
      -7.661205291748046875,
      7.4718637466430664062,
      9
    ));

    debug(validate_model(nn, dim, validation_n));
    // local output
    //   => 122: validate_model(nn, dim, validation_n)=0.879938
    // codingame output
    //   => 690: validate_model(nn, dim, validation_n)=0.879938
  }

}
