#include "mini_test.hpp"
#include "sovereign/mps_io.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/engine.hpp"
#include "sovereign/verifier.hpp"
#include <stdexcept>
using namespace sovereign;

namespace {
bool rejects(const std::string& text, const MpsParseOptions& options = {}) {
  try { load_model_from_mps_string(text, options); return false; }
  catch (const std::exception&) { return true; }
}
const std::string prefix = "ROWS\n N OBJ\n L CAP\nCOLUMNS\n";
}

TEST(Mps, MarkersIncludeFirstColumnAndStopBeforeNext) {
  auto m=load_model_from_mps_string(prefix+
    " M0 'MARKER' 'INTORG'\n X OBJ 5 CAP 1\n M1 'MARKER' 'INTEND'\n Y OBJ 2 CAP 1\nRHS\n R CAP 2\nENDATA\n").model;
  EXPECT_EQ(m.problem_type,ProblemType::MILP);
  EXPECT_EQ(m.variables[0].type,VariableType::Integer);
  EXPECT_EQ(m.variables[0].upper_bound,1.0);
  EXPECT_EQ(m.variables[1].type,VariableType::Continuous);
}

TEST(Mps, ObjectiveSelectionOffsetAndJsonRoundtrip) {
  auto m=load_model_from_mps_string("OBJSENSE\n MAX\nOBJNAME SECOND\nROWS\n N FIRST\n N SECOND\n L CAP\nCOLUMNS\n X FIRST 99 SECOND 3\n X CAP 1\nRHS\n R CAP 2 SECOND -7\nENDATA\n").model;
  EXPECT_EQ(m.sense,Sense::Maximize);
  EXPECT_EQ(m.objective.linear.at("X"),3.0);
  EXPECT_EQ(m.objective.constant,7.0);
  m=load_model_from_json_string(model_to_json_string(m));
  auto r=OptimizationEngine().solve(m);
  EXPECT_EQ(r.status,SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value,13.0,1e-8);
  EXPECT_TRUE(SolutionVerifier().verify(m,r).is_valid);
}

TEST(Mps, NamedRangesExpandSafelyAndChooseFirstSet) {
  auto m=load_model_from_mps_string(prefix+" X OBJ 1 CAP 1\nRHS\n FIRST CAP 8\n SECOND CAP 99\nRANGES\n RNG CAP 3\n OTHER CAP 100\nENDATA\n").model;
  EXPECT_EQ(m.constraints.size(),2u);
  EXPECT_EQ(m.constraints[0].sense,ConstraintSense::Le);
  EXPECT_EQ(m.constraints[0].rhs,8.0);
  EXPECT_EQ(m.constraints[1].sense,ConstraintSense::Ge);
  EXPECT_EQ(m.constraints[1].rhs,5.0);
  EXPECT_NEAR(OptimizationEngine().solve(m).objective_value,5.0,1e-8);
}

TEST(Mps, OmittedSetNamesAndNegativeUpperBound) {
  auto m=load_model_from_mps_string(prefix+" X OBJ 1 CAP 1\nRHS\n CAP -2\nBOUNDS\n UP X -1\nENDATA\n").model;
  EXPECT_EQ(m.constraints[0].rhs,-2.0);
  EXPECT_EQ(m.variables[0].lower_bound,-1e30);
  EXPECT_EQ(m.variables[0].upper_bound,-1.0);
}

TEST(Mps, BoundsOnlyVariablesAndFirstBoundSet) {
  auto m=load_model_from_mps_string(prefix+" X OBJ 1 CAP 1\nBOUNDS\n FX B Y 2\n UP B X 4\n UP OTHER X 99\nENDATA\n").model;
  EXPECT_EQ(m.variables.size(),2u);
  EXPECT_EQ(m.variables[0].upper_bound,4.0);
  EXPECT_EQ(m.variables[1].lower_bound,2.0);
}

TEST(Mps, FixedNamesAndBlankColumnContinuation) {
  auto card=[](const std::string& col,const std::string& row,const std::string& value) {
    std::string s(36,' '); s.replace(4,col.size(),col); s.replace(14,row.size(),row); s.replace(24,value.size(),value); return s+"\n";
  };
  auto m=load_model_from_mps_string("ROWS\n N  OBJ NAME\n L  CAP NAME\nCOLUMNS\n"+
    card("X NAME","OBJ NAME","2D+0")+card("","CAP NAME","1")+
    "RHS\n"+card("R","CAP NAME","3")+"ENDATA\n").model;
  EXPECT_EQ(m.variables[0].name,"X NAME");
  EXPECT_EQ(m.objective.linear.at("X NAME"),2.0);
  EXPECT_EQ(m.constraints[0].rhs,3.0);
}

TEST(Mps, DuplicateCoefficientsAreSummed) {
  auto m=load_model_from_mps_string(prefix+" X OBJ 1 CAP 2\n X OBJ 2 CAP -1\nENDATA\n").model;
  EXPECT_EQ(m.objective.linear.at("X"),3.0);
  EXPECT_EQ(m.constraints[0].linear.at("X"),1.0);
}

TEST(Mps, RejectsSilentCorruptionAndResourceOverruns) {
  EXPECT_TRUE(rejects(prefix+" X OBJ nan\nENDATA\n"));
  EXPECT_TRUE(rejects(prefix+" X OBJ 1\nBOUNDS\n SC B X 3\nENDATA\n"));
  EXPECT_TRUE(rejects(prefix+" X OBJ 1\n Y OBJ 2\n X CAP 1\nENDATA\n"));
  EXPECT_TRUE(rejects(prefix+" M 'MARKER' 'INTORG'\n X OBJ 1\nENDATA\n"));
  EXPECT_TRUE(rejects(prefix+" X OBJ 1\n"));
  MpsParseOptions o; o.max_nonzeros=1;
  EXPECT_TRUE(rejects(prefix+" X OBJ 1 CAP 2\nENDATA\n",o));
}

TEST(Mps, PerSolveAlgorithmAndPresolveOptions) {
  auto model=load_model_from_mps_string(prefix+" X OBJ -1 CAP 1\nRHS\n R CAP 3\nENDATA\n").model;
  EngineOptions options; options.presolve=false; options.lp_algorithm="simplex";
  auto simplex=OptimizationEngine().solve(model,options);
  EXPECT_EQ(simplex.status,SolverStatus::Optimal);
  EXPECT_NEAR(simplex.objective_value,-3,1e-8);
  EXPECT_TRUE(simplex.message.find("simplex")!=std::string::npos);
  options.lp_algorithm="ipm";
  auto ipm=OptimizationEngine().solve(model,options);
  EXPECT_EQ(ipm.status,SolverStatus::Optimal);
  EXPECT_NEAR(ipm.objective_value,-3,1e-7);
  EXPECT_TRUE(ipm.message.find("interior")!=std::string::npos);
}

TEST(Mps, FractionalIntegerRelaxationWithConstant) {
  auto model=load_model_from_mps_string("OBJSENSE MAX\n"+prefix+
    " M0 'MARKER' 'INTORG'\n X OBJ 5 CAP 2\n Y OBJ 4 CAP 2\n M1 'MARKER' 'INTEND'\nRHS\n R CAP 3 OBJ -10\nENDATA\n").model;
  auto result=OptimizationEngine().solve(model);
  EXPECT_EQ(result.status,SolverStatus::Optimal);
  EXPECT_NEAR(result.objective_value,15,1e-8);
  EXPECT_TRUE(SolutionVerifier().verify(model,result).is_valid);
}
