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
    total_prob += nn.forward_softmax(input)[target];
  }
  double avg_prob = total_prob / n;
  return avg_prob;
}

Nn train_model(int dim)
{
  srand(1234);
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
        nn.train_softmax(inputs, labels, 0.01);
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
    // 112: validate_model(nn, dim, validation_n)=0.870565
  }else{
    Nn nn = nn_eda::NnIo::from_obj(nn_eda::NnIo::Obj(
      std::vector<int>({5, 25, 25, 5, }),
      "㈩㗇㈠牋姰㔟狌娔餬燋稧띤烋겢㌞狌婘㔣狋觗맳燈㨌䄚饎㨃뤸看뙊㜛珌㔓뭄覎誰㜠먡訠㜠犉㨔㜚瀋뙊㌡珋뙉뚺犊뙱뤽猋㨌䌞牍騐㜙浦䨕㼞㽿뙊㬤犋娷묽牋骠㠠犌舏뤷燌㨣뵁犊騿띁猋橘㬠犋騿봽牋뙂㤥瞌婇㔛犌娈㔣熌橠㜠牌稷봿狋訨㤟痌娯謯熌騨㜠猌樧뤽獋訸䄠熌娠㴞爋騯餫牋穗뭃犋驏묻燌橘㔤犌樠㌢犌樨㜢牌䨔㤡灌樀㌟同樿猬爋驀㬢犑詠㌢爌㥬㔝炋稸㔚牋穇봽烋騘㬢职穥洧牌槈㜝猋樸㌞王樈㜜猋稸㜡狉뙁键焋稨㌞熋뙂㌞爌䨳띎狌䩃땀猌娯봽爋稯봽煋똹輳獋簘㔠烌娯띁狌穨㤤牌樕꼿烍䨬㤜珌㨫洿牋똙雵猌㐤㜢燋驷띀澋똲㔝堌樷紬牌䨬㤢爐詠㔛爌㥬㌟煌樧물烋똩봼瀌䨴㔝澋稐㼠獋騐㌝灌娰䄗熌娟焿犋槯鳶犌㐬㌥爋稯뤾獌娿봾猌樟뵁燋託뤾熋뙡륃獌橈㔝烋橐㌡牌祀㔠熋驀㌠爋뙑뵀犌䨄䜢爋뙂㜡澋騷꼽犋詐㜜熌稰㜞炌稟댿熊槇댻牋詀㜡浌䨓뭂燌婰㜘纻驈㔠燌㩻굁燋昷봽焌樟땁狋뚊㜨爋詀鴩燋駿넼牌䨫봽炌稨㜝爌䩌㔚皆稨㔡牌娨㤢牌娸㌣獌訰㬥狌婏뤿熋똲㬣牊뗚㴢珌㨤㤡獋樰㔣獋驰㬗划槸㴝玌稯뭂熌䨳뭁猌穈㌤牌䨜㔞犌䨜㤢獋穏뤼羋婀䌞燌㧳뵀燿詨㤦爋뚚㤡玌穗묾狋詇봿狌穗륂犋驈㤢獌穐㬤獌㧴㔘狋詈㜠牋稨㤤庌䩴霯獌똡뤽狊樿띄爋驈㼒燌訨䌝珌䩋봿濌婘㌨睂뛒愹狊䨴㔠悏䫄夼爋娧륌瘍뛊仿瑌機锱爌娯댾王婨㜧猋騨㌤燋똩륉砎蟠活碂㫜强灋槟굤眎蟈愼熌䩌䄜碎諮费眎諦덋琊䪄㬞忎竘愿爌詀㔡犌䨬㔡帉㦼㑺㭀䦗氫䷊艀㓰犴鑏㤟뉞佨㖩琋騰㔙立䩼伮猋樗뭋浊讀鬫珎뙚䤡瘎櫀嫨㈠",
      -7.6804871559143066406,
      7.6024932861328125,
      9
    ));

    debug(validate_model(nn, dim, validation_n));
    // local output
    //   => 123: validate_model(nn, dim, validation_n)=0.869781
    // codingame output
    //   => 723: validate_model(nn, dim, validation_n)=0.869781
  }

}
