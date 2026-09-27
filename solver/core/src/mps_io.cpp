#include "sovereign/mps_io.hpp"
#include "sovereign/json_io.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace sovereign {
namespace {
constexpr double inf = 1e30;
std::string trim(const std::string& s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
std::string upper(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}
std::vector<std::string> tokens(const std::string& s) {
  std::istringstream in(s);
  std::vector<std::string> out;
  for (std::string t; in >> t;) out.push_back(t);
  return out;
}
double number(std::string s) {
  const auto original = s;
  for (auto& c : s) if (c == 'd' || c == 'D') c = 'E';
  for (std::size_t i=1; i+1<s.size(); ++i) {
    if ((s[i]=='+' || s[i]=='-') && std::isdigit(static_cast<unsigned char>(s[i-1]))) {
      s.insert(i, "E"); break;
    }
  }
  std::size_t used=0;
  double v=std::stod(s, &used);
  if (used!=s.size() || !std::isfinite(v)) throw std::runtime_error("invalid number: " + original);
  return v;
}
std::string field(const std::string& s, std::size_t start, std::size_t length) {
  return start < s.size() ? trim(s.substr(start,length)) : "";
}
bool blank(const std::string& s, std::size_t start, std::size_t length) {
  return field(s,start,length).empty();
}
// Recognize the fixed card separators before preserving embedded spaces in names.
// Otherwise use free records. The option allows callers to force free parsing.
std::vector<std::string> fields(const std::string& s, const std::string& section, bool fixed) {
  if (fixed && s.size()>=5 && blank(s,3,1) && blank(s,12,2)) {
    if (section=="ROWS" && blank(s,0,1) && field(s,1,2).size()==1)
      return {field(s,1,2),field(s,4,8)};
    if (s.size()>=22 && blank(s,22,2) && blank(s,36,3) && blank(s,47,2)) {
      if (section=="BOUNDS") {
        std::vector<std::string> f={field(s,1,2),field(s,4,8),field(s,14,8)};
        if (!field(s,24,12).empty()) f.push_back(field(s,24,12));
        const auto type=upper(f[0]);
        const bool needs=type!="FR" && type!="MI" && type!="PL" && type!="BV";
        if ((!needs || f.size()==4) && blank(s,36,s.size())) return f;
        return tokens(s);
      }
      if (section=="COLUMNS" || section=="RHS" || section=="RANGES") {
        if (field(s,14,8)=="'MARKER'") return {field(s,4,8),"'MARKER'",field(s,39,8)};
        std::vector<std::string> f={field(s,4,8),field(s,14,8),field(s,24,12)};
        if (!field(s,39,8).empty()) { f.push_back(field(s,39,8)); f.push_back(field(s,49,12)); }
        bool valid=true;
        try { number(f[2]); if (f.size()==5) number(f[4]); }
        catch (const std::exception&) { valid=false; }
        if (valid) return f;
      }
    }
  }
  return tokens(s);
}
}

MpsParseResult load_model_from_mps_string(const std::string& text, const MpsParseOptions& options) {
  MpsParseResult out;
  auto& model=out.model;
  struct Row { std::string name; char type; double rhs=0, range=0; bool ranged=false; };
  std::vector<Row> rows;
  std::unordered_map<std::string,std::size_t> row_id, col_id;
  std::vector<std::unordered_map<std::string,double>> entries;
  std::vector<bool> bounded;
  std::string section, objective, requested_objective, last_column;
  std::string rhs_set, range_set, bound_set;
  bool rhs_seen=false, range_seen=false, bound_seen=false;
  bool int_block=false, saw_rows=false, saw_columns=false, ended=false;
  std::size_t line=0, entry_count=0;
  auto add_column=[&](const std::string& name) {
    if (name.empty()) throw std::runtime_error("missing column name");
    auto it=col_id.find(name);
    if (it!=col_id.end()) return it->second;
    if (model.variables.size()>=options.max_variables) throw std::runtime_error("variable limit exceeded");
    auto i=model.variables.size(); col_id[name]=i;
    Variable v; v.name=name; model.variables.push_back(v);
    entries.emplace_back(); bounded.push_back(false); return i;
  };
  auto select_set=[](const std::string& name, std::string& selected, bool& seen) {
    if (!seen) { selected=name; seen=true; }
    return name==selected;
  };
  std::istringstream in(text);
  for (std::string raw; std::getline(in,raw);) {
    ++line;
    try {
      if (line==1 && raw.compare(0,3,"\xEF\xBB\xBF")==0) raw.erase(0,3);
      if (trim(raw).empty() || raw[0]=='*' || trim(raw)[0]=='$') continue;
      auto comment=raw.find(" $"); if (comment!=std::string::npos) raw.resize(comment);
      if (!std::isspace(static_cast<unsigned char>(raw[0]))) {
        auto f=tokens(raw); auto head=upper(f[0]);
        if (head=="ENDATA") { ended=true; break; }
        if (head!="NAME" && head!="ROWS" && head!="COLUMNS" && head!="RHS" && head!="RANGES" && head!="BOUNDS" && head!="OBJSENSE" && head!="OBJNAME")
          throw std::runtime_error("unsupported MPS section: " + head);
        section=head;
        if (head=="ROWS") saw_rows=true;
        if (head=="COLUMNS") saw_columns=true;
        if ((head=="OBJSENSE" || head=="OBJNAME") && f.size()>1) {
          if (head=="OBJNAME") requested_objective=f[1];
          else {
            auto sense=upper(f[1]);
            if (sense!="MAX" && sense!="MIN" && sense!="MAXIMIZE" && sense!="MINIMIZE") throw std::runtime_error("invalid OBJSENSE");
            model.sense=sense.substr(0,3)=="MAX" ? Sense::Maximize : Sense::Minimize;
          }
          section="NAME";
        }
        continue;
      }
      auto f=fields(raw,section,options.fixed_format);
      if (f.empty()) continue;
      if (section=="OBJSENSE" || section=="OBJNAME") {
        if (section=="OBJNAME") requested_objective=trim(raw);
        else {
          auto sense=upper(f[0]);
          if (sense!="MAX" && sense!="MIN" && sense!="MAXIMIZE" && sense!="MINIMIZE") throw std::runtime_error("invalid OBJSENSE");
          model.sense=sense.substr(0,3)=="MAX" ? Sense::Maximize : Sense::Minimize;
        }
        section="NAME";
      } else if (section=="ROWS") {
        if (f.size()!=2 || f[0].size()!=1 || f[1].empty()) throw std::runtime_error("malformed ROWS record");
        char type=upper(f[0])[0];
        if (std::string("NLEG").find(type)==std::string::npos) throw std::runtime_error("invalid row type");
        if (row_id.count(f[1])) throw std::runtime_error("duplicate row: " + f[1]);
        if (rows.size()>=options.max_constraints) throw std::runtime_error("row limit exceeded");
        row_id[f[1]]=rows.size(); rows.push_back({f[1],type});
        if (type=='N' && objective.empty()) objective=f[1];
      } else if (section=="COLUMNS") {
        auto marker=std::find(f.begin(),f.end(),"'MARKER'");
        if (marker!=f.end()) {
          if (marker+1==f.end()) throw std::runtime_error("missing integer marker kind");
          auto kind=*(marker+1);
          if (kind=="'INTORG'" && !int_block) int_block=true;
          else if (kind=="'INTEND'" && int_block) int_block=false;
          else throw std::runtime_error("unmatched or nested integer marker");
          last_column.clear(); continue;
        }
        if (f.size()<3 || f.size()%2!=1) throw std::runtime_error("malformed COLUMNS record");
        if (f[0].empty()) f[0]=last_column;
        if (col_id.count(f[0]) && last_column!=f[0]) throw std::runtime_error("noncontiguous column: " + f[0]);
        auto c=add_column(f[0]); last_column=f[0];
        if (int_block) model.variables[c].type=VariableType::Integer;
        for (std::size_t j=1;j<f.size();j+=2) {
          if (!row_id.count(f[j])) throw std::runtime_error("unknown row: " + f[j]);
          if (++entry_count>options.max_nonzeros) throw std::runtime_error("nonzero limit exceeded");
          entries[c][f[j]]+=number(f[j+1]);
          if (!std::isfinite(entries[c][f[j]])) throw std::runtime_error("coefficient overflow");
        }
      } else if (section=="RHS" || section=="RANGES") {
        std::size_t start=f.size()%2; std::string set=start ? f[0] : "";
        if (f.size()-start<2) throw std::runtime_error("malformed row/value pairs");
        bool selected=section=="RHS" ? select_set(set,rhs_set,rhs_seen) : select_set(set,range_set,range_seen);
        if (!selected) continue;
        for (auto j=start;j<f.size();j+=2) {
          if (!row_id.count(f[j])) throw std::runtime_error("unknown row: " + f[j]);
          auto& r=rows[row_id.at(f[j])]; auto value=number(f[j+1]);
          if (section=="RHS") r.rhs=value;
          else { r.range=value; r.ranged=true; }
        }
      } else if (section=="BOUNDS") {
        auto type=upper(f[0]); bool value=type!="FR" && type!="MI" && type!="PL" && type!="BV";
        std::size_t expected=value ? 4 : 3;
        if (f.size()==expected-1) f.insert(f.begin()+1,"");
        if (f.size()!=expected) throw std::runtime_error("malformed BOUNDS record");
        if (!select_set(f[1],bound_set,bound_seen)) continue;
        auto c=add_column(f[2]); auto& v=model.variables[c]; double n=value ? number(f[3]) : 0;
        if (type=="UP" || type=="UI") { v.upper_bound=n; if (!bounded[c] && n<0) v.lower_bound=-inf; }
        else if (type=="LO" || type=="LI") v.lower_bound=n;
        else if (type=="FX") v.lower_bound=v.upper_bound=n;
        else if (type=="FR") { v.lower_bound=-inf; v.upper_bound=inf; }
        else if (type=="MI") v.lower_bound=-inf;
        else if (type=="PL") v.upper_bound=inf;
        else if (type=="BV") { v.type=VariableType::Binary; v.lower_bound=0; v.upper_bound=1; }
        else throw std::runtime_error("unsupported bound type: " + type + " (semi-continuous bounds require an explicit reformulation)");
        if (type=="LI" || type=="UI") v.type=VariableType::Integer;
        bounded[c]=true;
      } else throw std::runtime_error("unexpected data record");
    } catch (const std::exception& e) { throw std::runtime_error("MPS line " + std::to_string(line) + ": " + e.what()); }
  }
  if (!saw_rows || !saw_columns || !ended) throw std::runtime_error("MPS requires ROWS, COLUMNS and ENDATA");
  if (int_block) throw std::runtime_error("unterminated INTORG block");
  if (!requested_objective.empty()) objective=requested_objective;
  if (!objective.empty() && (!row_id.count(objective) || rows[row_id.at(objective)].type!='N')) throw std::runtime_error("OBJNAME must select an N row");
  for (std::size_t c=0;c<model.variables.size();++c) {
    auto& v=model.variables[c];
    if (v.type==VariableType::Integer && !bounded[c]) v.upper_bound=1;
    if (v.lower_bound>v.upper_bound) throw std::runtime_error("crossed bounds: " + v.name);
    if (v.type!=VariableType::Continuous) model.problem_type=ProblemType::MILP;
    auto it=entries[c].find(objective);
    if (it!=entries[c].end()) model.objective.linear[v.name]=it->second;
  }
  if (!objective.empty()) model.objective.constant=-rows[row_id.at(objective)].rhs;
  std::vector<std::unordered_map<std::string,double>> row_entries(rows.size());
  for (std::size_t j=0;j<entries.size();++j)
    for (const auto& e:entries[j])
      if (e.second!=0) row_entries[row_id.at(e.first)][model.variables[j].name]=e.second;
  for (auto& r:rows) {
    if (r.type=='N') continue;
    Constraint c; c.name=r.name; c.rhs=r.rhs;
    c.sense=r.type=='L' ? ConstraintSense::Le : r.type=='G' ? ConstraintSense::Ge : ConstraintSense::Eq;
    c.linear=std::move(row_entries[row_id.at(r.name)]);
    if (r.ranged) {
      double lo=r.rhs, hi=r.rhs;
      if (r.type=='G' || (r.type=='E' && r.range>=0)) hi+=std::abs(r.range);
      else lo-=std::abs(r.range);
      c.sense=ConstraintSense::Le; c.rhs=hi;
      model.constraints.push_back(c);
      c.name+="_lo"; while (row_id.count(c.name)) c.name+="_lo";
      c.sense=ConstraintSense::Ge; c.rhs=lo;
    }
    model.constraints.push_back(c);
    if (model.constraints.size()>options.max_constraints) throw std::runtime_error("expanded row limit exceeded");
  }
  out.rows=model.constraints.size(); out.columns=model.variables.size(); out.objective_name=objective;
  for (auto& c:model.constraints) out.nonzeros+=c.linear.size();
  if (out.nonzeros>options.max_nonzeros) throw std::runtime_error("expanded nonzero limit exceeded");
  return out;
}

MpsParseResult load_model_from_mps_file(const std::string& path,
                                       const MpsParseOptions& options) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("could not open MPS file: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  MpsParseResult r = load_model_from_mps_string(ss.str(), options);
  if (!r.objective_name.empty()) {
    r.notes.push_back("objective row: " + r.objective_name);
  }
  return r;
}

OptimizationModel load_model_from_file(const std::string& path) {
  const std::size_t dot = path.find_last_of('.');
  const std::string ext =
      dot == std::string::npos ? std::string() : [&] {
        std::string e = path.substr(dot + 1);
        std::transform(e.begin(), e.end(), e.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return e;
      }();

  if (ext == "mps") return load_model_from_mps_file(path).model;
  if (ext == "json") return load_model_from_json_file(path);
  // Compressed variants are not supported yet; say so rather than producing a
  // JSON parse error, which tells the user nothing useful.
  if (ext == "gz" || ext == "bz2" || ext == "zip") {
    throw std::runtime_error("compressed model files (" + ext + ") are not supported yet; " +
                             "decompress to .mps or .json first");
  }
  if (ext.empty()) {
    throw std::runtime_error("cannot infer the model format: '" + path +
                             "' has no extension. Use .mps or .json.");
  }
  throw std::runtime_error("unsupported model extension '." + ext +
                           "'. Supported: .mps, .json");
}

}  // namespace sovereign
