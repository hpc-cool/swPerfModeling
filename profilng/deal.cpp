#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <set>
#include <map>
#include <sys/stat.h>
#include <string.h>
using namespace std;

typedef long long ll;
typedef struct 
{
	string self,father;
    ll times,acctime,alltime;
}node;



int main(int argc, char * argv[])
{
    cout<<"begin"<<endl;
	ifstream fr2;
	fr2.open("./out/func.out.tmp");
	
	string line,inter;
    vector<string> v;
    map<string,ll>selftime;
    map<string,ll>::iterator it;
    ll tmp=0;
	while(getline(fr2,line))
	{
		for (int i = 0; i < line.length(); ++i)
		{
			if (line[i] == ',')
			{
				line[i] = ' ';
			}
		}
		stringstream is(line);
        v.clear();
		while (is >> inter)
		{
			v.push_back(inter);
		}

        string fa=v[1],son=v[0];
    cout<<"1"<<endl;

        ll acctime=stoll(v[3]),alltime=stoll(v[4]);
        selftime[son]+=acctime;
        selftime[fa]-=alltime;
        tmp+=alltime-acctime;
        cout<<fa<<","<<son<<","<<1.0*tmp/1000000<<endl;
	}
	fr2.close();

	ofstream fw;
    fw.open("./out/selftime.out");
    int linenum=0;
    ll outtime=0;
    for(it=selftime.begin();it!=selftime.end();it++)
    {
        fw<<linenum<<","<<it->first<<","<<it->second<<endl;
        outtime+=it->second>0?it->second:0;
        linenum++;
    }
    fw<<"all time: "<<1.0*outtime/1000000<<" ms"<<endl;
    fw<<"error time: "<<1.0*tmp/1000000<<" ms"<<endl;
    fw.close();
	
	return 0;
}