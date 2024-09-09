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

double validate_model(Inferrer &nn, int dim, int n)
{
  srand(22222);

  double total_prob = 0.0;
  rep(loop, n){
    int target = rand() % dim;
    vector<float> input = gen_input(dim, target);
    nn.forward(input);
    total_prob += nn.get_prediction(0, target);
  }
  double avg_prob = total_prob / n;
  return avg_prob;
}

Trainer train_model(int dim)
{
  srand(1234);
  Trainer nn({dim, dim * 5, dim * 5}, {dim, dim});
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
        vector<vector<vector<float>>> labels;
        rep(i, batch_size){
          int target = rand() % dim;
          inputs.push_back(gen_input(dim, target));
          labels.push_back({
            gen_label(dim, target),
            gen_label(dim, target)
          });
        }
        nn.train(inputs, labels, 0.01);
      }
    }
  }

  return nn;
}

int main()
{
  // 1. Run this code with -D_LOCAL, which exports a model to codingame_nn.txt.
  // 2. Copy and paste the epxorted model from codingame_nn.txt to this code.
  // 3. Run this code without -D_LOCAL.
  // 4. Submit this code to codingame along with nn.cpp.
  // 5. Confirm the outputs from 3 and 4 exactly match.

  int dim = 5;
  int validation_n = 10000;
  #ifdef LOCAL
  bool is_local = true;
  #else
  bool is_local = false;
  #endif

  debug(is_local);

  if(is_local){
    Trainer nn = train_model(dim);
    {
      ofstream ofs("codingame_nn.txt");
      ofs << NnIo::serialize(nn, 9);
    }
    debug(validate_model(nn, dim, validation_n));
    // 112: validate_model(nn, dim, validation_n)=0.870565
  }else{
    Inferrer inferrer = nn_eda::NnIo::deserialize("3 5 25 25 2 5 5 -11.140069007873535156 11.030337333679199219 9 ㈩㙉㈠爋뙊㔣熋牷뤼琌䨣묾獋臧뜽爌穀㌠燌騇뭀熌樨㌟뒞㨓뭅玎䨤儩熌䎃묻璌战㔠熌䨤㌠牋橀㜠王稟뜿燌㩣렾煋㨛밐畘뙂㜞珌樠㜗㺁騟뜽滋詘㜞燋稿뵀熌娰㈡獋爏묾珋䨫띀燋詏뤽犌䩄㜤狋똱묿牌䨬㴢牌稘㌠燌稰㜢牌䨄㬝狌詀㔤犌䨜㤡爌㨴㔢狌訸㜠煍뙠㼧烌樐㔠獌㨌㬣煍䗼㔞牌䨬㜡犋뙙봼爌橐㔣犌娨㔢犌娨㜢牌䨜㜡玍娨㴨烌娧뤼玊뙉덄磃婐㌠灹娷뤼狌䨬㌤熌䨫봾猋뙢㜢珋뙱뵁牊詈㤞猌騸㔧熌娏묻獋娧륀璈婇넽熂똪㌟熍婘㙥狋觷륌獌婏蔰犍䨳뜽澌騯餫燌䨓묿牌䨼㜞爋뙂㔠牌㨼㌠狌樠㔡浍橀㤣犌樗묿吋樯꺷琍뙡륎爌稟댾焂蔰㴩煍뙚㌞狌䧫륀獍婘㔝熌㨤㌟牌樸㌡燋뙂㔠爌䨬㴡犌婈㜟狋騰㜞泍娠㜤燌䨛봿压騷뚵燍婈㌰犋騧뤽猌䨳봾狌娟뵀爋騟묿燌㨻뭁猌婀㔞犌樈㜟爌驐㜣犌㨬㔣牌㨬㴡牌訟묿狍驐㬧燌䩃봼珊訧띅䭶詗뜻燍騰㔟犌㦻륀狋橇묽爋뙊㤠燌䨛봽狋㨣봿烌똲㑩犋뗩륍玌娿缲狍㨫땀濌䨓錪狌姈㌞犍穏묾狌樸㔠犌㩄㬡猋똱봿牌䨤㜢牌䨬㌢猌稰㤣狌婇묿燋똺㤣猌㨓뭀狌䨼㬞獋騯뵁珌㩋뭃狍䨤㜞獌娧봿燌䨣뵁狌橀㔢牋騨㌟猌㨜㔢玃㫄㬥燸秨㔠玌䨃땃燍詗뜼爋뙊㬠獌橏묿犋騿봿犌橇뭁犋뙚㜢猌橈㤣灎娾㬥焋詐㔢珋騐㜦煍瘀㔟燌㨫딾燋訿뭁燋訿뭁牍㧬㤡犍䧴㬢瑈䩜䤪瑈㩔䤩煋橷묺爋骐㌞皋稐㌚看騘㜛牌䨼㜠燌㨴㌞瓈穸䬨璈婨䤦猋騰㔢獋뙊㜣猋詈㜠犌䨳뭁沍䩄䌦沍䩄䌦爋똡뭈牌㨃뭈燌㨴㤢燌䨴㬣焌橈㜤烌婀㔣玌姨㤦猌䧌㔤燋똙륈燋똙륈瓌뚉茴璌詯脲爋訠䬞牋訠䬟燋骘㌞焋橿뤹獋穀㤣猋穀㜢玍婠䔊獍䩌䌉猌䨬㔡獌婈㜢甍穯脳瑍䩃礰牌㨻봿牋詇봽猋稧봿王驀㔣犋뙚㔣狌樰㤣犌䨬㔡燋扨㑧漁删㐚䃀剨㔖뛛承輬琌䄬㐟瑌䭬굾熑娈㌘縋똲㭔稌䯛넹狒똺䬠橒듪鬐橒듪鬐㈠");

    debug(validate_model(inferrer, dim, validation_n));
    // local output
    //   => 124: validate_model(nn, dim, validation_n)=0.859775
    // codingame output
    //   => 750: validate_model(nn, dim, validation_n)=0.859775
  }

}
